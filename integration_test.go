package openwakeword

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

func TestOfficialONNXModels(t *testing.T) {
	library := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	models := os.Getenv("OPENWAKEWORD_MODEL_DIR")
	if library == "" || models == "" {
		t.Skip("set ONNXRUNTIME_SHARED_LIBRARY_PATH and OPENWAKEWORD_MODEL_DIR to run integration test")
	}
	ort.SetSharedLibraryPath(library)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatalf("initialize ONNX Runtime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			t.Fatalf("destroy ONNX Runtime: %v", err)
		}
	}()
	engine, err := New(Options{
		MelspectrogramModelPath: filepath.Join(models, "melspectrogram.onnx"),
		EmbeddingModelPath:      filepath.Join(models, "embedding_model.onnx"),
		VADModelPath:            filepath.Join(models, "silero_vad.onnx"),
		VADThreshold:            0.5,
		WakeWordModels: []ModelConfig{{
			Path: filepath.Join(models, "alexa_v0.1.onnx"), Name: "alexa",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	var scores map[string]float32
	for i := 0; i < 30; i++ {
		scores, err = engine.Predict(make([]int16, FrameSamples))
		if err != nil {
			t.Fatal(err)
		}
	}
	if score, ok := scores["alexa"]; !ok || score < 0 || score > 1 {
		t.Fatalf("invalid Alexa prediction: %v", scores)
	}
	if scores["alexa"] != 0 {
		t.Fatalf("VAD should suppress silence, got %v", scores)
	}
	if wav := os.Getenv("OPENWAKEWORD_TEST_WAV"); wav != "" {
		engine.Reset()
		predictions, err := engine.PredictWAV(wav, FrameSamples, time.Second, PredictOptions{})
		if err != nil {
			t.Fatal(err)
		}
		var maximum float32
		for _, prediction := range predictions {
			if prediction["alexa"] > maximum {
				maximum = prediction["alexa"]
			}
		}
		if maximum < 0.5 {
			t.Fatalf("official Alexa clip was not detected; max score %.5f", maximum)
		}
	}
}
