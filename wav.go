package openwakeword

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"time"
)

type WAV struct {
	SampleRate int
	Channels   int
	Samples    []int16
}

func ReadWAV(path string) (wav WAV, err error) {
	file, err := os.Open(path)

	if err != nil {
		return WAV{}, err
	}

	defer func(file *os.File) {
		if deferErr := file.Close(); deferErr != nil {
			err = errors.Join(err, deferErr)
		}
	}(file)

	return DecodeWAV(file)
}

func DecodeWAV(r io.Reader) (WAV, error) {
	header := make([]byte, 12)
	if _, err := io.ReadFull(r, header); err != nil {
		return WAV{}, fmt.Errorf("read WAV header: %w", err)
	}
	if string(header[:4]) != "RIFF" || string(header[8:]) != "WAVE" {
		return WAV{}, errors.New("input is not a RIFF/WAVE file")
	}
	var format uint16
	var channels uint16
	var sampleRate uint32
	var bits uint16
	var data []byte
	for {
		chunk := make([]byte, 8)
		if _, err := io.ReadFull(r, chunk); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				break
			}
			return WAV{}, err
		}
		size := binary.LittleEndian.Uint32(chunk[4:])
		payload := make([]byte, size)
		if _, err := io.ReadFull(r, payload); err != nil {
			return WAV{}, fmt.Errorf("read WAV chunk %q: %w", chunk[:4], err)
		}
		if size%2 == 1 {
			var padding [1]byte
			_, _ = io.ReadFull(r, padding[:])
		}
		switch string(chunk[:4]) {
		case "fmt ":
			if len(payload) < 16 {
				return WAV{}, errors.New("invalid WAV fmt chunk")
			}
			format = binary.LittleEndian.Uint16(payload[0:2])
			channels = binary.LittleEndian.Uint16(payload[2:4])
			sampleRate = binary.LittleEndian.Uint32(payload[4:8])
			bits = binary.LittleEndian.Uint16(payload[14:16])
		case "data":
			data = payload
		}
	}
	if format != 1 || channels != 1 || sampleRate != SampleRate || bits != 16 {
		return WAV{}, fmt.Errorf("openWakeWord requires mono 16-bit PCM at 16000 Hz; got format=%d channels=%d rate=%d bits=%d",
			format, channels, sampleRate, bits)
	}
	if len(data)%2 != 0 {
		return WAV{}, errors.New("WAV data has an odd byte count")
	}
	samples := make([]int16, len(data)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(data[i*2:]))
	}
	return WAV{SampleRate: int(sampleRate), Channels: int(channels), Samples: samples}, nil
}

func (e *Engine) PredictWAV(path string, chunkSize int, padding time.Duration, opts PredictOptions) ([]map[string]float32, error) {
	wav, err := ReadWAV(path)
	if err != nil {
		return nil, err
	}
	return e.PredictClip(wav.Samples, chunkSize, padding, opts)
}

func (e *Engine) PredictClip(samples []int16, chunkSize int, padding time.Duration, opts PredictOptions) ([]map[string]float32, error) {
	if chunkSize <= 0 {
		chunkSize = FrameSamples
	}
	paddingSamples := int(float64(SampleRate) * padding.Seconds())
	data := make([]int16, 0, paddingSamples*2+len(samples))
	data = append(data, make([]int16, paddingSamples)...)
	data = append(data, samples...)
	data = append(data, make([]int16, paddingSamples)...)
	predictions := make([]map[string]float32, 0, (len(data)+chunkSize-1)/chunkSize)
	for start := 0; start < len(data); start += chunkSize {
		end := start + chunkSize
		if end > len(data) {
			end = len(data)
		}
		prediction, err := e.PredictWithOptions(data[start:end], opts)
		if err != nil {
			return nil, err
		}
		predictions = append(predictions, prediction)
	}
	return predictions, nil
}
