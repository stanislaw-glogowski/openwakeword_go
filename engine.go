package openwakeword

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const predictionHistory = 30

type AudioFilter interface {
	Process([]int16) ([]int16, error)
	Reset()
}

type Verifier interface {
	Score(features []float32, shape []int64) (float32, error)
}

type ModelConfig struct {
	Path         string
	Name         string
	ClassMapping map[int]string
	Verifier     Verifier
}

type Options struct {
	MelspectrogramModelPath string
	EmbeddingModelPath      string
	WakeWordModels          []ModelConfig
	VADModelPath            string
	VADThreshold            float32
	VerifierThreshold       float32
	AudioFilter             AudioFilter
}

type PredictOptions struct {
	Patience     map[string]int
	Threshold    map[string]float32
	DebounceTime time.Duration
}

type wakeWordModel struct {
	name        string
	session     *onnxSession
	inputFrames int
	outputCount int
	classes     map[int]string
	labels      []string
	verifier    Verifier
}

// Engine is a streaming openWakeWord detector. Calls to Predict are safe to
// make from multiple goroutines, though audio is processed in call order.
//
// The caller owns the global github.com/yalue/onnxruntime_go environment. Call
// ort.SetSharedLibraryPath and ort.InitializeEnvironment before constructing an
// Engine, and call ort.DestroyEnvironment after closing all Engines and VADs.
type Engine struct {
	mu                sync.Mutex
	audioFeatures     *AudioFeatures
	models            []*wakeWordModel
	history           map[string][]float32
	filter            AudioFilter
	vad               *VAD
	verifierThreshold float32
	closed            bool
}

func New(opts Options) (*Engine, error) {
	if opts.MelspectrogramModelPath == "" || opts.EmbeddingModelPath == "" {
		return nil, errors.New("melspectrogram and embedding model paths are required")
	}
	audioFeatures, err := NewAudioFeatures(opts.MelspectrogramModelPath, opts.EmbeddingModelPath)
	if err != nil {
		return nil, err
	}
	e := &Engine{
		audioFeatures:     audioFeatures,
		history:           make(map[string][]float32),
		filter:            opts.AudioFilter,
		verifierThreshold: opts.VerifierThreshold,
	}
	if e.verifierThreshold == 0 {
		e.verifierThreshold = 0.1
	}
	for _, config := range opts.WakeWordModels {
		if _, err := e.addModel(config); err != nil {
			_ = e.Close()
			return nil, err
		}
	}
	if opts.VADThreshold > 0 {
		if opts.VADModelPath == "" {
			_ = e.Close()
			return nil, errors.New("VAD model path is required when VADThreshold is enabled")
		}
		e.vad, err = NewVAD(opts.VADModelPath, opts.VADThreshold)
		if err != nil {
			_ = e.Close()
			return nil, err
		}
	}
	return e, nil
}

// NewModel mirrors the Python package's primary constructor name.
func NewModel(opts Options) (*Engine, error) { return New(opts) }

// NewEngine builds an Engine from already constructed feature and VAD objects.
// It is useful when the caller wants to share or customize those components.
func NewEngine(vad *VAD, audioFeatures *AudioFeatures) (*Engine, error) {
	if audioFeatures == nil {
		return nil, errors.New("audio features are required")
	}
	return &Engine{
		vad:               vad,
		audioFeatures:     audioFeatures,
		history:           make(map[string][]float32),
		verifierThreshold: 0.1,
	}, nil
}

func (e *Engine) AddModel(path string) (string, error) {
	if e == nil {
		return "", errors.New("engine is nil")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	model, err := e.addModel(ModelConfig{Path: path})
	if err != nil {
		return "", err
	}
	return model.name, nil
}

func (e *Engine) AddNamedModel(name, path string) error {
	if e == nil {
		return errors.New("engine is nil")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, err := e.addModel(ModelConfig{Name: name, Path: path})
	return err
}

func (e *Engine) AddModelConfig(config ModelConfig) error {
	if e == nil {
		return errors.New("engine is nil")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, err := e.addModel(config)
	return err
}

func (e *Engine) addModel(config ModelConfig) (*wakeWordModel, error) {
	if e == nil || e.closed {
		return nil, errors.New("engine is closed")
	}
	if strings.ToLower(filepath.Ext(config.Path)) != ".onnx" {
		return nil, fmt.Errorf("only ONNX wake-word models are supported, got %q", filepath.Ext(config.Path))
	}
	name := config.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(config.Path), filepath.Ext(config.Path))
	}
	for _, existing := range e.models {
		if existing.name == name {
			return nil, fmt.Errorf("model %q is already loaded", name)
		}
	}
	session, err := newONNXSession(config.Path)
	if err != nil {
		return nil, fmt.Errorf("load wake-word model %q: %w", name, err)
	}
	if len(session.inputs) != 1 || len(session.inputs[0].Dimensions) < 2 || session.inputs[0].Dimensions[1] <= 0 {
		_ = session.close()
		return nil, fmt.Errorf("wake-word model %q has unsupported input shape", name)
	}
	if len(session.outputs) != 1 {
		_ = session.close()
		return nil, fmt.Errorf("wake-word model %q must have one output", name)
	}
	outputCount := lastPositiveDimension(session.outputs[0].Dimensions)
	if outputCount <= 0 {
		_ = session.close()
		return nil, fmt.Errorf("wake-word model %q has unsupported output shape", name)
	}
	classes := config.ClassMapping
	if len(classes) == 0 {
		classes = make(map[int]string, outputCount)
		if outputCount == 1 {
			classes[0] = name
		} else {
			for i := 0; i < outputCount; i++ {
				classes[i] = fmt.Sprint(i)
			}
		}
	}
	labels := make([]string, 0, len(classes))
	for index, label := range classes {
		if index < 0 || index >= outputCount || label == "" {
			_ = session.close()
			return nil, fmt.Errorf("invalid class mapping %d=%q for model %q", index, label, name)
		}
		labels = append(labels, label)
	}
	sort.Strings(labels)
	model := &wakeWordModel{
		name:        name,
		session:     session,
		inputFrames: int(session.inputs[0].Dimensions[1]),
		outputCount: outputCount,
		classes:     classes,
		labels:      labels,
		verifier:    config.Verifier,
	}
	e.models = append(e.models, model)
	return model, nil
}

func (e *Engine) Predict(samples []int16) (map[string]float32, error) {
	return e.PredictWithOptions(samples, PredictOptions{})
}

func (e *Engine) PredictWithOptions(samples []int16, opts PredictOptions) (map[string]float32, error) {
	if e == nil {
		return nil, errors.New("engine is nil")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, errors.New("engine is closed")
	}
	if len(e.models) == 0 {
		return nil, errors.New("no wake-word models loaded")
	}
	if len(opts.Patience) > 0 && opts.DebounceTime > 0 {
		return nil, errors.New("patience and debounce cannot be used together")
	}
	if (len(opts.Patience) > 0 || opts.DebounceTime > 0) && len(opts.Threshold) == 0 {
		return nil, errors.New("thresholds are required with patience or debounce")
	}
	processedSamples := samples
	if e.filter != nil {
		var err error
		processedSamples, err = e.filter.Process(samples)
		if err != nil {
			return nil, fmt.Errorf("filter audio: %w", err)
		}
	}
	prepared, err := e.audioFeatures.Process(processedSamples)
	if err != nil {
		return nil, err
	}
	predictions := make(map[string]float32)
	for _, model := range e.models {
		for _, label := range model.labels {
			predictions[label] = e.latest(label)
		}
		if prepared == 0 {
			continue
		}
		maxScores := make([]float32, model.outputCount)
		for offset := prepared/FrameSamples - 1; offset >= 0; offset-- {
			features, shape, err := e.audioFeatures.Features(model.inputFrames, offset)
			if err != nil {
				return nil, err
			}
			values, _, err := model.session.runFloat(shape, features)
			if err != nil {
				return nil, fmt.Errorf("predict with %q: %w", model.name, err)
			}
			if len(values) < model.outputCount {
				return nil, fmt.Errorf("model %q returned %d scores, expected %d", model.name, len(values), model.outputCount)
			}
			for i := range maxScores {
				if values[i] > maxScores[i] {
					maxScores[i] = values[i]
				}
			}
		}
		features, featureShape, _ := e.audioFeatures.Features(model.inputFrames, 0)
		for index, label := range model.classes {
			score := maxScores[index]
			if model.verifier != nil && score >= e.verifierThreshold {
				score, err = model.verifier.Score(features, featureShape)
				if err != nil {
					return nil, fmt.Errorf("verify %q: %w", label, err)
				}
			}
			if len(e.history[label]) < 5 {
				score = 0
			}
			predictions[label] = score
		}
	}

	for label, score := range predictions {
		parent := e.parentModel(label)
		threshold, hasThreshold := opts.Threshold[parent]
		if patience := opts.Patience[parent]; patience > 0 && score != 0 {
			if !hasThreshold {
				return nil, fmt.Errorf("missing threshold for model %q", parent)
			}
			if countAtLeast(tail(e.history[label], patience), threshold) < patience {
				predictions[label] = 0
			}
		} else if opts.DebounceTime > 0 && hasThreshold && score >= threshold && prepared > 0 {
			frameDuration := time.Duration(float64(time.Second) * float64(prepared) / SampleRate)
			n := int(math.Ceil(float64(opts.DebounceTime) / float64(frameDuration)))
			if countAtLeast(tail(e.history[label], n), threshold) > 0 {
				predictions[label] = 0
			}
		}
	}
	for label, score := range predictions {
		e.history[label] = appendHistory(e.history[label], score)
	}
	if e.vad != nil {
		if _, err := e.vad.Predict(samples, 640); err != nil {
			return nil, err
		}
		if !e.vad.CheckContextScore() {
			for label := range predictions {
				predictions[label] = 0
			}
		}
	}
	return predictions, nil
}

func (e *Engine) Reset() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.history = make(map[string][]float32)
	if e.audioFeatures != nil {
		e.audioFeatures.Reset()
	}
	if e.filter != nil {
		e.filter.Reset()
	}
	if e.vad != nil {
		e.vad.Reset()
	}
}

func (e *Engine) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	var errs []error
	for _, model := range e.models {
		errs = append(errs, model.session.close())
	}
	if e.vad != nil {
		errs = append(errs, e.vad.Close())
	}
	if e.audioFeatures != nil {
		errs = append(errs, e.audioFeatures.Close())
	}
	return errors.Join(errs...)
}

func (e *Engine) parentModel(label string) string {
	for _, model := range e.models {
		for _, candidate := range model.classes {
			if candidate == label {
				return model.name
			}
		}
	}
	return label
}

func (e *Engine) latest(label string) float32 {
	h := e.history[label]
	if len(h) == 0 {
		return 0
	}
	return h[len(h)-1]
}

func lastPositiveDimension(shape []int64) int {
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] > 0 {
			return int(shape[i])
		}
	}
	return 0
}

func appendHistory(history []float32, score float32) []float32 {
	history = append(history, score)
	if len(history) > predictionHistory {
		history = history[len(history)-predictionHistory:]
	}
	return history
}

func tail(values []float32, n int) []float32 {
	if n <= 0 || len(values) <= n {
		return values
	}
	return values[len(values)-n:]
}

func countAtLeast(values []float32, threshold float32) int {
	n := 0
	for _, value := range values {
		if value >= threshold {
			n++
		}
	}
	return n
}
