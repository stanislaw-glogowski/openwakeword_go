package openwakeword

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultModelThreshold    = 0.9
	defaultModelPatience     = 0
	defaultModelDebounceTime = 0
	predictionHistory        = 30
)

type (
	// ModelOptions configures how a wake-word model score is interpreted.
	ModelOptions struct {
		Name         string
		Threshold    float32
		Patience     int
		DebounceTime time.Duration
	}
	model struct {
		name         string
		session      *onnxSession
		threshold    float32
		patience     int
		debounceTime time.Duration
		inputFrames  int
		history      []float32
	}
)

// WithModelName sets the result key used for a wake-word model.
func WithModelName(name string) func(*ModelOptions) {
	return func(opts *ModelOptions) {
		opts.Name = name
	}
}

// WithModelThreshold sets the score threshold used by patience and debounce.
func WithModelThreshold(threshold float32) func(*ModelOptions) {
	return func(opts *ModelOptions) {
		opts.Threshold = threshold
	}
}

// WithModelPatience requires the model history to stay above the threshold for
// the given number of frames before scores are allowed through.
func WithModelPatience(patience int) func(*ModelOptions) {
	return func(opts *ModelOptions) {
		opts.Patience = patience
	}
}

// WithModelDebounce suppresses repeated detections within the given duration.
func WithModelDebounce(debounceTime time.Duration) func(*ModelOptions) {
	return func(opts *ModelOptions) {
		opts.DebounceTime = debounceTime
	}
}

func buildModelOptions(options ...func(*ModelOptions)) ModelOptions {
	opts := &ModelOptions{
		Threshold:    defaultModelThreshold,
		Patience:     defaultModelPatience,
		DebounceTime: defaultModelDebounceTime,
	}
	for _, opt := range options {
		opt(opts)
	}
	return *opts
}

func newModel(path string, opts ModelOptions) (*model, error) {
	if opts.Threshold == 0 {
		opts.Threshold = defaultModelThreshold
	}
	if opts.Patience == 0 {
		opts.Patience = defaultModelPatience
	}
	if opts.DebounceTime == 0 {
		opts.DebounceTime = defaultModelDebounceTime
	}
	if strings.ToLower(filepath.Ext(path)) != ".onnx" {
		return nil, fmt.Errorf("wake-word model must be an ONNX file, got %q", filepath.Ext(path))
	}
	name := opts.Name
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	if opts.Patience > 0 && opts.DebounceTime > 0 {
		return nil, errors.New("model patience and debounce cannot be used together")
	}
	if (opts.Patience > 0 || opts.DebounceTime > 0) && opts.Threshold == 0 {
		return nil, errors.New("model threshold is required with patience or debounce")
	}
	session, err := newONNXSession(path)
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
	if outputCount != 1 {
		_ = session.close()
		return nil, fmt.Errorf("wake-word model %q must return one score, got %d", name, outputCount)
	}
	return &model{
		name:         name,
		session:      session,
		threshold:    opts.Threshold,
		patience:     opts.Patience,
		debounceTime: opts.DebounceTime,
		inputFrames:  int(session.inputs[0].Dimensions[1]),
	}, nil
}

func (m *model) latest() float32 {
	if len(m.history) == 0 {
		return 0
	}
	return m.history[len(m.history)-1]
}

func (m *model) reset() {
	m.history = m.history[:0]
}

func (m *model) close() error {
	return m.session.close()
}

func (m *model) appendHistory(score float32) {
	m.history = append(m.history, score)
	if len(m.history) > predictionHistory {
		m.history = m.history[len(m.history)-predictionHistory:]
	}
}

func (m *model) countAtLeast(n int) (count int) {
	values := m.history
	if n > 0 && len(values) > n {
		values = values[len(values)-n:]
	}
	for _, value := range values {
		if value >= m.threshold {
			count++
		}
	}
	return
}

func lastPositiveDimension(shape []int64) int {
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] > 0 {
			return int(shape[i])
		}
	}
	return 0
}
