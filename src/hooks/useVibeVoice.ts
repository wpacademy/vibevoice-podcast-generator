/**
 * useVibeVoice - React hook for VibeVoice TTS streaming
 *
 * @example
 * ```tsx
 * const { read, stop, isReading, isConnecting, error } = useVibeVoice(
 *   { api: "http://localhost:8888/api" },
 *   { model: "microsoft/VibeVoice-Realtime-0.5B" },
 *   { speaker_name: "en-Emma_woman" },
 *   { device: "cuda" }
 * );
 *
 * // Start reading text
 * await read("Hello, world!");
 *
 * // Stop the current reading
 * stop();
 * ```
 */

import { useCallback, useEffect, useRef, useState } from "react";

// Configuration types
interface ApiConfig {
  api: string;
}

interface ModelConfig {
  model?: string;
}

interface SpeakerConfig {
  speaker_name?: string;
}

interface DeviceConfig {
  device?: "cuda" | "cpu" | "mps";
}

interface SynthesisOptions {
  cfg_scale?: number;
  inference_steps?: number;
}

interface VibeVoiceState {
  isReading: boolean;
  isConnecting: boolean;
  error: string | null;
  progress: number;
}

interface VibeVoiceReturn extends VibeVoiceState {
  read: (text: string, options?: SynthesisOptions) => Promise<void>;
  stop: () => void;
}

interface LogMessage {
  type: "log" | "error";
  event: string;
  data?: Record<string, unknown>;
  message?: string;
  timestamp: string;
}

const SAMPLE_RATE = 24_000;

/**
 * React hook for streaming TTS from VibeVoice API
 */
export function useVibeVoice(
  apiConfig: ApiConfig,
  _modelConfig?: ModelConfig,
  speakerConfig?: SpeakerConfig,
  _deviceConfig?: DeviceConfig,
): VibeVoiceReturn {
  const [state, setState] = useState<VibeVoiceState>({
    isReading: false,
    isConnecting: false,
    error: null,
    progress: 0,
  });

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<AudioBuffer[]>([]);
  const isPlayingRef = useRef(false);
  const nextPlayTimeRef = useRef(0);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  /**
   * Initialize or get AudioContext
   */
  const getAudioContext = useCallback((): AudioContext => {
    if (!audioContextRef.current || audioContextRef.current.state === "closed") {
      audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
    }
    return audioContextRef.current;
  }, []);

  /**
   * Convert PCM16 bytes to AudioBuffer
   */
  const pcm16ToAudioBuffer = useCallback(
    (pcmData: ArrayBuffer): AudioBuffer => {
      const ctx = getAudioContext();
      const int16Array = new Int16Array(pcmData);
      const float32Array = new Float32Array(int16Array.length);

      // Convert PCM16 to Float32
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
      }

      const audioBuffer = ctx.createBuffer(1, float32Array.length, SAMPLE_RATE);
      audioBuffer.getChannelData(0).set(float32Array);

      return audioBuffer;
    },
    [getAudioContext],
  );

  /**
   * Schedule audio buffer for playback
   */
  const scheduleAudioPlayback = useCallback(
    (audioBuffer: AudioBuffer) => {
      const ctx = getAudioContext();

      if (ctx.state === "suspended") {
        ctx.resume();
      }

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);

      // Track active sources for cleanup
      activeSourcesRef.current.add(source);
      source.onended = () => {
        activeSourcesRef.current.delete(source);
      };

      const currentTime = ctx.currentTime;
      const startTime = Math.max(currentTime, nextPlayTimeRef.current);

      source.start(startTime);
      nextPlayTimeRef.current = startTime + audioBuffer.duration;
    },
    [getAudioContext],
  );

  /**
   * Process audio queue
   */
  const processAudioQueue = useCallback(() => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) {
      return;
    }

    isPlayingRef.current = true;

    while (audioQueueRef.current.length > 0) {
      const buffer = audioQueueRef.current.shift();
      if (buffer) {
        scheduleAudioPlayback(buffer);
      }
    }

    isPlayingRef.current = false;
  }, [scheduleAudioPlayback]);

  /**
   * Stop playback and cleanup
   */
  const stop = useCallback(() => {
    // Close WebSocket connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Stop all active audio sources
    activeSourcesRef.current.forEach((source) => {
      try {
        source.stop();
        source.disconnect();
      } catch {
        // Source may have already ended
      }
    });
    activeSourcesRef.current.clear();

    // Clear audio queue
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    nextPlayTimeRef.current = 0;

    setState((prev) => ({
      ...prev,
      isReading: false,
      isConnecting: false,
      progress: 0,
    }));
  }, []);

  /**
   * Start reading text with TTS
   */
  const read = useCallback(
    async (text: string, options?: SynthesisOptions): Promise<void> => {
      // Stop any existing playback
      stop();

      if (!text.trim()) {
        setState((prev) => ({ ...prev, error: "Text cannot be empty" }));
        return;
      }

      return new Promise((resolve, reject) => {
        setState((prev) => ({
          ...prev,
          isConnecting: true,
          error: null,
          progress: 0,
        }));

        // Build WebSocket URL
        const baseUrl = apiConfig.api.replace(/^http/, "ws");
        const wsUrl = new URL(`${baseUrl}/stream`);
        wsUrl.searchParams.set("text", text);

        if (speakerConfig?.speaker_name) {
          wsUrl.searchParams.set("voice", speakerConfig.speaker_name);
        }
        if (options?.cfg_scale !== undefined) {
          wsUrl.searchParams.set("cfg", options.cfg_scale.toString());
        }
        if (options?.inference_steps !== undefined) {
          wsUrl.searchParams.set("steps", options.inference_steps.toString());
        }

        const ws = new WebSocket(wsUrl.toString());
        wsRef.current = ws;

        // Reset playback timing
        const ctx = getAudioContext();
        nextPlayTimeRef.current = ctx.currentTime;

        ws.onopen = () => {
          setState((prev) => ({
            ...prev,
            isConnecting: false,
            isReading: true,
          }));
        };

        ws.onmessage = (event) => {
          if (typeof event.data === "string") {
            // JSON log message
            try {
              const message: LogMessage = JSON.parse(event.data);
              //   console.log("[VibeVoice]", message.event, message.data);

              if (message.type === "error") {
                setState((prev) => ({
                  ...prev,
                  error: message.message || message.event,
                }));
              }

              if (message.event === "model_progress" && message.data) {
                const progress = message.data.generated_sec as number;
                setState((prev) => ({ ...prev, progress }));
              }
            } catch {
              console.warn("[VibeVoice] Failed to parse message:", event.data);
            }
          } else if (event.data instanceof Blob) {
            // Binary audio data
            event.data.arrayBuffer().then((buffer) => {
              const audioBuffer = pcm16ToAudioBuffer(buffer);
              audioQueueRef.current.push(audioBuffer);
              processAudioQueue();
            });
          }
        };

        ws.onerror = (error) => {
          console.error("[VibeVoice] WebSocket error:", error);
          setState((prev) => ({
            ...prev,
            isConnecting: false,
            isReading: false,
            error: "WebSocket connection error",
          }));
          reject(new Error("WebSocket connection error"));
        };

        ws.onclose = (event) => {
          wsRef.current = null;

          if (event.code === 1013) {
            // Server busy
            setState((prev) => ({
              ...prev,
              isConnecting: false,
              isReading: false,
              error: "Server is busy. Please try again.",
            }));
            reject(new Error("Server is busy"));
          } else {
            setState((prev) => ({
              ...prev,
              isConnecting: false,
              isReading: false,
            }));
            resolve();
          }
        };
      });
    },
    [
      apiConfig.api,
      speakerConfig?.speaker_name,
      stop,
      getAudioContext,
      pcm16ToAudioBuffer,
      processAudioQueue,
    ],
  );

  return {
    read,
    stop,
    isReading: state.isReading,
    isConnecting: state.isConnecting,
    error: state.error,
    progress: state.progress,
  };
}

export default useVibeVoice;
