package openwakeword

import (
	"errors"
	"fmt"
	"slices"
)

const (
	defaultVADFrameSize = 640
	defaultVADThreshold = 0.5
)

type (
	// VADOptions configures Silero VAD scoring.
	VADOptions struct {
		FrameSize int
		Threshold float32
	}

	// VAD wraps a stateful Silero ONNX VAD model.
	VAD struct {
		opts    VADOptions
		session *onnxSession
		h       []float32
		c       []float32
		history Samples
	}
)

// WithVADFrameSize sets how many samples are scored per VAD model call.
func WithVADFrameSize(frameSize int) func(*VADOptions) {
	return func(opts *VADOptions) {
		opts.FrameSize = frameSize
	}
}

// WithVADThreshold sets the speech score threshold used by Detect.
func WithVADThreshold(threshold float32) func(*VADOptions) {
	return func(opts *VADOptions) {
		opts.Threshold = threshold
	}
}

// NewVAD loads a Silero VAD model using functional options.
func NewVAD(path string, options ...func(*VADOptions)) (*VAD, error) {
	opts := &VADOptions{}
	for _, opt := range options {
		opt(opts)
	}
	return NewVADWithOptions(path, *opts)
}

// NewVADWithOptions loads a Silero VAD model using a VADOptions value directly.
func NewVADWithOptions(path string, opts VADOptions) (*VAD, error) {
	if opts.Threshold == 0 {
		opts.Threshold = defaultVADThreshold
	}
	if opts.FrameSize == 0 {
		opts.FrameSize = defaultVADFrameSize
	}
	if opts.FrameSize <= 0 {
		return nil, errors.New("vad frame size must be positive")
	}
	model, err := newONNXSession(path)
	if err != nil {
		return nil, fmt.Errorf("load vad model: %w", err)
	}
	v := &VAD{session: model, opts: opts}
	v.Reset()
	return v, nil
}

// Predict scores the provided samples and updates the recurrent VAD state.
func (v *VAD) Predict(samples Samples) (float32, error) {
	if len(samples) == 0 {
		return 0, nil
	}
	var sum float32
	var chunks int
	for start := 0; start < len(samples); start += v.opts.FrameSize {
		end := start + v.opts.FrameSize
		if end > len(samples) {
			end = len(samples)
		}
		audio := slices.Clone(samples[start:end])
		score, h, c, err := v.session.runVAD(audio, v.h, v.c, SampleRate)
		if err != nil {
			return 0, err
		}
		v.h = append(v.h[:0], h...)
		v.c = append(v.c[:0], c...)
		sum += score
		chunks++
	}
	average := sum / float32(chunks)
	v.history = append(v.history, average)
	if len(v.history) > 125 {
		v.history = v.history[len(v.history)-125:]
	}
	return average, nil
}

// Detect reports whether the current samples exceed the VAD threshold.
func (v *VAD) Detect(samples Samples) (bool, error) {
	score, err := v.Predict(samples)
	if err != nil {
		return false, err
	}
	return score > v.opts.Threshold, nil
}

// DetectContext updates VAD state and applies openWakeWord's delayed speech
// context check.
func (v *VAD) DetectContext(samples Samples) (bool, error) {
	_, err := v.Predict(samples)
	if err != nil {
		return false, err
	}
	return v.contextScore() > v.opts.Threshold, nil
}

// Reset clears recurrent state and score history.
func (v *VAD) Reset() {
	v.h = make([]float32, 2*64)
	v.c = make([]float32, 2*64)
	v.history = v.history[:0]
}

// Close releases the VAD ONNX session.
func (v *VAD) Close() error {
	if v == nil || v.session == nil {
		return nil
	}
	err := v.session.close()
	v.session = nil
	return err
}

func (v *VAD) contextScore() float32 {
	end := len(v.history) - 4
	start := len(v.history) - 7
	if start < 0 || end <= start {
		return 0
	}
	score := v.history[start]
	for _, s := range v.history[start+1 : end] {
		if s > score {
			score = s
		}
	}
	return score
}
