package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/gordonklaus/portaudio"
	oww "github.com/stanislaw-glogowski/openwakeword_go"
	ort "github.com/yalue/onnxruntime_go"
)

type modelConfig struct {
	name string
	path string
}

type vadConfig struct {
	path      string
	threshold float64
}

type config struct {
	threshold        float64
	debounce, warmup time.Duration
	runtimePath,
	melspecPath,
	embeddingPath string
	vad    *vadConfig
	models []modelConfig
}

func (c config) vadThreshold() float64 {
	if c.vad == nil {
		return 0
	}
	return c.vad.threshold
}

func main() {
	log.SetFlags(log.Ltime | log.Lmicroseconds)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	cfg, err := readConfig()
	if err != nil {
		log.Fatalf("read config: %v", err)
	}

	ort.SetSharedLibraryPath(cfg.runtimePath)

	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("ort.InitializeEnvironment: %v", err)
	}

	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Fatalf("ort.DestroyEnvironment: %v", err)
		}
	}()

	audioFeatures, err := oww.NewAudioFeatures(cfg.melspecPath, cfg.embeddingPath)
	if err != nil {
		log.Fatalf("openwakeword.NewAudioFeatures: %v", err)
	}
	var vad *oww.VAD
	if cfg.vad != nil {
		vad, err = oww.NewVAD(cfg.vad.path, oww.WithVADThreshold(float32(cfg.vad.threshold)))
		if err != nil {
			log.Fatalf("openwakeword.NewVAD: %v", err)
		}
	}
	engine, err := oww.New(audioFeatures, vad)
	if err != nil {
		log.Fatalf("openwakeword.New: %v", err)
	}
	for _, model := range cfg.models {
		if err := engine.AddModel(model.path, oww.WithModelName(model.name)); err != nil {
			log.Fatalf("engine.AddModel: %v", err)
		}
	}

	defer func() {
		if err := engine.Close(); err != nil {
			log.Fatalf("engine.Close: %v", err)
		}
	}()

	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("portaudio.Initialize: %v", err)
	}

	defer func() {
		if err := portaudio.Terminate(); err != nil {
			log.Fatalf("portaudio.Terminate: %v", err)
		}
	}()

	input := make(oww.Samples, oww.FrameSamples)
	stream, err := portaudio.OpenDefaultStream(
		1,
		0,
		float64(oww.SampleRate),
		len(input),
		input,
	)

	if err != nil {
		log.Fatalf("portaudio.OpenDefaultStream: %v", err)
	}

	defer func() {
		if err := stream.Close(); err != nil {
			log.Fatalf("stream.Close: %v", err)
		}
	}()

	if err := stream.Start(); err != nil {
		log.Fatalf("stream.Start: %v", err)
	}

	defer func() {
		if err := stream.Stop(); err != nil {
			log.Fatalf("stream.Stop: %v", err)
		}
	}()

	readyAt := time.Now().Add(cfg.warmup)

	log.Printf("warming up microphone and models for %s", cfg.warmup)

	readyLogged := false

	lastDetections := make(map[string]time.Time)

	for ctx.Err() == nil {
		if err := stream.Read(); err != nil {
			if errors.Is(err, portaudio.InputOverflowed) {
				log.Printf("stream.Read: microphone input overflow; continuing")
				continue
			}
			log.Fatalf("stream.Read: %v", err)
		}

		scores, err := engine.Predict(input)
		if err != nil {
			log.Fatalf("engine.Predict: %v", err)
		}

		if time.Now().Before(readyAt) {
			continue
		}

		if !readyLogged {
			log.Printf("listening for wake word; threshold=%.2f, VAD=%.2f, press Ctrl+C to stop", cfg.threshold, cfg.vadThreshold())
			readyLogged = true
		}

		for modelName, score := range scores {
			if lastDetection := lastDetections[modelName]; float64(score) >= cfg.threshold && time.Since(lastDetection) >= cfg.debounce {
				log.Printf("wake word detected; score=%.2f, model=%v", score, modelName)
				lastDetections[modelName] = time.Now()
			}
		}
	}
}

func readConfig() (cfg config, err error) {
	var runtimePath,
		melspecPath,
		embeddingPath string
	var vad *vadConfig
	var models []modelConfig

	defaultCwd, err := os.Getwd()

	if err != nil {
		return config{}, fmt.Errorf("get current working directory: %w", err)
	}

	cwd := flag.String("cwd", defaultCwd, "current working directory")
	threshold := flag.Float64("threshold", 0.9, "wake-word detection threshold from 0 to 1")
	warmup := flag.Duration("warmup", 3*time.Second, "ignore detections while microphone and models stabilize")
	debounce := flag.Duration("debounce", 1500*time.Millisecond, "minimum delay between detections")
	vadThreshold := flag.Float64("vad-threshold", 0.5, "Silero speech threshold from 0 to 1; 0 disables VAD")

	flag.Parse()

	runtimePath = filepath.Join(*cwd, "runtime", onnxRuntimeLibraryName())
	modelsDir := filepath.Join(*cwd, "models")
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return config{}, fmt.Errorf("read models directory %q: %w", modelsDir, err)
	}

	for _, entry := range entries {
		if entry.IsDir() || strings.ToLower(filepath.Ext(entry.Name())) != ".onnx" {
			continue
		}
		path := filepath.Join(modelsDir, entry.Name())
		name := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		switch name {
		case "melspectrogram":
			melspecPath = path
		case "embedding_model":
			embeddingPath = path
		case "silero_vad":
			vad = &vadConfig{path: path, threshold: *vadThreshold}
		default:
			models = append(models, modelConfig{name: name, path: path})
		}
	}
	sort.Slice(models, func(i, j int) bool {
		return models[i].name < models[j].name
	})

	if err := requireRegularFile(runtimePath, "ONNX Runtime library"); err != nil {
		return config{}, err
	}
	if err := requireRegularFile(melspecPath, "mel spectrogram model"); err != nil {
		return config{}, err
	}
	if err := requireRegularFile(embeddingPath, "embedding model"); err != nil {
		return config{}, err
	}
	if len(models) == 0 {
		return config{}, fmt.Errorf("no wake-word ONNX models found in %q", modelsDir)
	}
	for _, model := range models {
		if err := requireRegularFile(model.path, fmt.Sprintf("wake-word model %q", model.name)); err != nil {
			return config{}, err
		}
	}
	if *vadThreshold > 0 {
		if vad == nil {
			return config{}, fmt.Errorf("vad threshold is enabled but %q is missing", filepath.Join(modelsDir, "silero_vad.onnx"))
		}
		if err := requireRegularFile(vad.path, "VAD model"); err != nil {
			return config{}, err
		}
	} else {
		vad = nil
	}

	return config{
		threshold:     *threshold,
		debounce:      *debounce,
		warmup:        *warmup,
		runtimePath:   runtimePath,
		melspecPath:   melspecPath,
		embeddingPath: embeddingPath,
		vad:           vad,
		models:        models,
	}, nil
}

func onnxRuntimeLibraryName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "libonnxruntime.so"
	}
}

func requireRegularFile(path, label string) error {
	if path == "" {
		return fmt.Errorf("%s path is required", label)
	}
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("%s does not exist: %s", label, path)
		}
		return fmt.Errorf("stat %s %q: %w", label, path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("%s must be a file, got directory: %s", label, path)
	}
	return nil
}
