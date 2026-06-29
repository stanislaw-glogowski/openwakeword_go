package openwakeword

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestPredictionHistoryHelpers(t *testing.T) {
	m := &model{threshold: 32, history: make([]float32, 0, defaultModelPredictionHistory+5)}
	for i := 0; i < defaultModelPredictionHistory+5; i++ {
		m.appendHistory(float32(i))
	}
	if len(m.history) != defaultModelPredictionHistory {
		t.Fatalf("expected %d history entries, got %d", defaultModelPredictionHistory, len(m.history))
	}
	if m.history[0] != 5 {
		t.Fatalf("expected oldest retained score 5, got %v", m.history[0])
	}
	if got := m.countAtLeast(4); got != 3 {
		t.Fatalf("expected 3 scores at least 32, got %d", got)
	}
}

func TestPredictionHistoryHelpersCustomLimit(t *testing.T) {
	m := &model{predictionHistory: 3, history: make([]float32, 0, 5)}
	for i := 0; i < 5; i++ {
		m.appendHistory(float32(i))
	}

	if len(m.history) != 3 {
		t.Fatalf("expected 3 history entries, got %d", len(m.history))
	}
	if m.history[0] != 2 {
		t.Fatalf("expected oldest retained score 2, got %v", m.history[0])
	}
}

func TestModelOptionsNormalize(t *testing.T) {
	opts := ModelOptions{}

	if err := opts.normalize(); err != nil {
		t.Fatal(err)
	}
	if opts.Threshold != defaultModelThreshold {
		t.Fatalf("Threshold = %v, want %v", opts.Threshold, defaultModelThreshold)
	}
	if opts.PredictionHistory != defaultModelPredictionHistory {
		t.Fatalf("PredictionHistory = %d, want %d", opts.PredictionHistory, defaultModelPredictionHistory)
	}
	if opts.Patience != defaultModelPatience {
		t.Fatalf("Patience = %d, want %d", opts.Patience, defaultModelPatience)
	}
	if opts.DebounceTime != defaultModelDebounceTime {
		t.Fatalf("DebounceTime = %v, want %v", opts.DebounceTime, defaultModelDebounceTime)
	}
}

func TestModelOptionsNormalizeInvalid(t *testing.T) {
	tests := []struct {
		name string
		opts ModelOptions
	}{
		{
			name: "negative prediction history",
			opts: ModelOptions{PredictionHistory: -1},
		},
		{
			name: "negative patience",
			opts: ModelOptions{Patience: -1},
		},
		{
			name: "negative debounce",
			opts: ModelOptions{DebounceTime: -1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.opts.normalize()
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestLastPositiveDimension(t *testing.T) {
	if got := lastPositiveDimension([]int64{-1, 1, 7}); got != 7 {
		t.Fatalf("expected 7, got %d", got)
	}
	if got := lastPositiveDimension([]int64{-1, -1}); got != 0 {
		t.Fatalf("expected 0, got %d", got)
	}
}

func TestDecodeWAV(t *testing.T) {
	samples := []int16{-32768, -1, 0, 1, 32767}
	var data bytes.Buffer
	data.WriteString("RIFF")
	_ = binary.Write(&data, binary.LittleEndian, uint32(36+len(samples)*2))
	data.WriteString("WAVEfmt ")
	_ = binary.Write(&data, binary.LittleEndian, uint32(16))
	_ = binary.Write(&data, binary.LittleEndian, uint16(1))
	_ = binary.Write(&data, binary.LittleEndian, uint16(1))
	_ = binary.Write(&data, binary.LittleEndian, uint32(SampleRate))
	_ = binary.Write(&data, binary.LittleEndian, uint32(SampleRate*2))
	_ = binary.Write(&data, binary.LittleEndian, uint16(2))
	_ = binary.Write(&data, binary.LittleEndian, uint16(16))
	data.WriteString("data")
	_ = binary.Write(&data, binary.LittleEndian, uint32(len(samples)*2))
	for _, sample := range samples {
		_ = binary.Write(&data, binary.LittleEndian, sample)
	}

	wav, err := DecodeWAV(bytes.NewReader(data.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if len(wav.Samples) != len(samples) || wav.Samples[0] != -1 || wav.Samples[2] != 0 || wav.Samples[4] != 1 {
		t.Fatalf("unexpected WAV samples: %+v", wav)
	}
}
