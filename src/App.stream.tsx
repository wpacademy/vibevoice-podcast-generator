import {
  Download,
  FileAudio,
  GripVertical,
  Loader2,
  Mic,
  Play,
  Plus,
  Square,
  Trash2,
  Upload,
  Volume2,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useVibeVoice } from "@/hooks/useVibeVoice";

const API_BASE = "http://localhost:8880/api";
const SAMPLE_RATE = 24_000;

interface ConfigResponse {
  voices: string[];
  default_voice: string;
  sample_rate: number;
  model: string;
  device: string;
}

interface PodcastSegment {
  id: string;
  text: string;
  voice: string;
}

interface PodcastData {
  segments: PodcastSegment[];
  createdAt: string;
  version: string;
}

const languageNames: Record<string, string> = {
  en: "🇬🇧 English",
  de: "🇩🇪 German",
  fr: "🇫🇷 French",
  it: "🇮🇹 Italian",
  sp: "🇪🇸 Spanish",
  pt: "🇵🇹 Portuguese",
  nl: "🇳🇱 Dutch",
  pl: "🇵🇱 Polish",
  jp: "🇯🇵 Japanese",
  kr: "🇰🇷 Korean",
  in: "🇮🇳 Indian English",
};

// Delay helper
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Generate audio for a single segment via WebSocket
 * Returns PCM16 audio data as Int16Array
 */
async function generateSegmentAudio(
  text: string,
  voice: string,
  onProgress?: (message: string) => void,
): Promise<Int16Array> {
  return new Promise((resolve, reject) => {
    const chunks: ArrayBuffer[] = [];

    const baseUrl = API_BASE.replace(/^http/, "ws");
    const wsUrl = new URL(`${baseUrl}/stream`);
    wsUrl.searchParams.set("text", text);
    wsUrl.searchParams.set("voice", voice);
    wsUrl.searchParams.set("cfg", "1.5");
    wsUrl.searchParams.set("steps", "5");

    const ws = new WebSocket(wsUrl.toString());

    ws.onopen = () => {
      onProgress?.("Generating...");
    };

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        event.data.arrayBuffer().then((buffer) => {
          chunks.push(buffer);
        });
      } else if (typeof event.data === "string") {
        try {
          const message = JSON.parse(event.data);
          if (message.event === "model_progress" && message.data) {
            onProgress?.(`${message.data.generated_sec?.toFixed(1)}s generated`);
          }
        } catch {
          // Ignore parse errors
        }
      }
    };

    ws.onerror = () => {
      reject(new Error("WebSocket connection error"));
    };

    ws.onclose = (event) => {
      if (event.code === 1013) {
        reject(new Error("Server is busy"));
        return;
      }

      // Combine all chunks into a single Int16Array
      const totalLength = chunks.reduce((acc, chunk) => acc + chunk.byteLength, 0);
      const combined = new Int16Array(totalLength / 2);
      let offset = 0;

      for (const chunk of chunks) {
        const int16 = new Int16Array(chunk);
        combined.set(int16, offset);
        offset += int16.length;
      }

      resolve(combined);
    };
  });
}

/**
 * Create WAV file from PCM16 data
 */
function createWavFile(samples: Int16Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;
  const fileSize = headerSize + dataSize;

  const buffer = new ArrayBuffer(fileSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, fileSize - 8, true);
  writeString(view, 8, "WAVE");

  // fmt chunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // AudioFormat (1 for PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // Write samples
  const dataView = new Int16Array(buffer, headerSize);
  dataView.set(samples);

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/**
 * Add silence between segments (in samples)
 */
function createSilence(durationMs: number): Int16Array {
  const numSamples = Math.floor((durationMs / 1000) * SAMPLE_RATE);
  return new Int16Array(numSamples);
}

/**
 * Podcast Maker - Create multi-voice podcasts with TTS
 */
export function PodcastMaker() {
  const [segments, setSegments] = useState<PodcastSegment[]>([
    {
      id: crypto.randomUUID(),
      text: "Welcome to our podcast! Today we'll be discussing exciting topics.",
      voice: "en-Emma_woman",
    },
  ]);

  const [voices, setVoices] = useState<string[]>([]);
  const [isLoadingVoices, setIsLoadingVoices] = useState(true);
  const [playingSegmentId, setPlayingSegmentId] = useState<string | null>(null);
  const [isPlayingAll, setIsPlayingAll] = useState(false);
  const [currentPlayingIndex, setCurrentPlayingIndex] = useState<number>(-1);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const stopPlayAllRef = useRef(false);
  const stopExportRef = useRef(false);
  const segmentPlayFunctionsRef = useRef<Map<string, () => Promise<void>>>(new Map());

  // Fetch available voices on mount
  useEffect(() => {
    const fetchVoices = async () => {
      try {
        const response = await fetch(`${API_BASE}/config`);
        if (response.ok) {
          const config: ConfigResponse = await response.json();
          setVoices(config.voices);
        }
      } catch (err) {
        console.error("Failed to fetch voices:", err);
        // Default voices if API is unavailable
        setVoices([
          "en-Carter_man",
          "en-Davis_man",
          "en-Emma_woman",
          "en-Frank_man",
          "en-Grace_woman",
          "en-Mike_man",
        ]);
      } finally {
        setIsLoadingVoices(false);
      }
    };

    fetchVoices();
  }, []);

  // Group voices by language
  const groupedVoices = voices.reduce<Record<string, string[]>>((acc, voice) => {
    const [lang] = voice.split("-");
    if (!acc[lang]) {
      acc[lang] = [];
    }
    acc[lang].push(voice);
    return acc;
  }, {});

  const addSegment = () => {
    const newSegment: PodcastSegment = {
      id: crypto.randomUUID(),
      text: "",
      voice: voices[0] || "en-Emma_woman",
    };
    setSegments([...segments, newSegment]);
  };

  const removeSegment = (id: string) => {
    if (segments.length > 1) {
      setSegments(segments.filter((s) => s.id !== id));
    }
  };

  const updateSegment = (id: string, field: keyof PodcastSegment, value: string) => {
    setSegments(segments.map((s) => (s.id === id ? { ...s, [field]: value } : s)));
  };

  const exportPodcastJson = () => {
    const data: PodcastData = {
      segments,
      createdAt: new Date().toISOString(),
      version: "1.0.0",
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `podcast-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importPodcast = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data: PodcastData = JSON.parse(e.target?.result as string);
        if (data.segments && Array.isArray(data.segments)) {
          // Regenerate IDs to force fresh component instances with new hooks
          // This ensures the voice from the imported JSON is used correctly
          setSegments(
            data.segments.map((s) => ({
              ...s,
              id: crypto.randomUUID(),
            })),
          );
        }
      } catch (err) {
        console.error("Failed to parse podcast file:", err);
        alert("Invalid podcast file format");
      }
    };
    reader.readAsText(file);

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Register segment play function
  const registerPlayFunction = useCallback((id: string, playFn: () => Promise<void>) => {
    segmentPlayFunctionsRef.current.set(id, playFn);
  }, []);

  // Unregister segment play function
  const unregisterPlayFunction = useCallback((id: string) => {
    segmentPlayFunctionsRef.current.delete(id);
  }, []);

  // Play all segments sequentially
  const handlePlayAll = async () => {
    setIsPlayingAll(true);
    stopPlayAllRef.current = false;

    for (let i = 0; i < segments.length; i++) {
      if (stopPlayAllRef.current) {
        break;
      }

      const segment = segments[i];
      if (!segment.text.trim()) {
        continue; // Skip empty segments
      }

      setCurrentPlayingIndex(i);
      setPlayingSegmentId(segment.id);

      const playFn = segmentPlayFunctionsRef.current.get(segment.id);
      if (playFn) {
        try {
          await playFn();
        } catch (err) {
          console.error(`Failed to play segment ${i + 1}:`, err);
        }
      }

      // Wait a bit before starting next segment to ensure server is ready
      if (i < segments.length - 1 && !stopPlayAllRef.current) {
        await delay(2500);
      }
    }

    setIsPlayingAll(false);
    setCurrentPlayingIndex(-1);
    setPlayingSegmentId(null);
  };

  const handleStopAll = () => {
    stopPlayAllRef.current = true;
    setIsPlayingAll(false);
    setCurrentPlayingIndex(-1);
    setPlayingSegmentId(null);
  };

  // Export podcast as audio file
  const handleExportAudio = async () => {
    const validSegments = segments.filter((s) => s.text.trim());
    if (validSegments.length === 0) {
      alert("No segments with text to export");
      return;
    }

    setIsExporting(true);
    stopExportRef.current = false;

    const audioChunks: Int16Array[] = [];
    // const silenceBetweenSegments = createSilence(500); // 500ms silence between segments
    const silenceBetweenSegments = createSilence(1); // 500ms silence between segments

    try {
      for (let i = 0; i < validSegments.length; i++) {
        if (stopExportRef.current) {
          setIsExporting(false);
          setExportProgress("");
          return;
        }

        const segment = validSegments[i];
        setExportProgress(`Generating segment ${i + 1}/${validSegments.length}...`);

        const audio = await generateSegmentAudio(segment.text, segment.voice, (msg) =>
          setExportProgress(`Segment ${i + 1}/${validSegments.length}: ${msg}`),
        );

        audioChunks.push(audio);

        // Add silence between segments (but not after the last one)
        if (i < validSegments.length - 1) {
          audioChunks.push(silenceBetweenSegments);
        }

        // Wait a bit before next segment
        if (i < validSegments.length - 1) {
          await delay(500);
        }
      }

      setExportProgress("Creating audio file...");

      // Combine all audio chunks
      const totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const combined = new Int16Array(totalLength);
      let offset = 0;

      for (const chunk of audioChunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }

      // Create WAV file and download
      const wavBlob = createWavFile(combined, SAMPLE_RATE);
      const url = URL.createObjectURL(wavBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `podcast-${Date.now()}.wav`;
      a.click();
      URL.revokeObjectURL(url);

      setExportProgress("Done!");
      await delay(1000);
    } catch (err) {
      console.error("Failed to export audio:", err);
      alert(`Failed to export audio: ${err instanceof Error ? err.message : "Unknown error"}`);
    }

    setIsExporting(false);
    setExportProgress("");
  };

  const handleCancelExport = () => {
    stopExportRef.current = true;
  };

  const isBusy = isPlayingAll || isExporting;

  return (
    <div className="min-h-screen bg-zinc-950">
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-10">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-3 bg-teal-600 rounded-xl">
              <Mic className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-3xl font-semibold text-white">Podcast Maker</h1>
          </div>
          <p className="text-zinc-500">Create multi-voice podcasts with AI text-to-speech</p>
        </header>

        {/* Action Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6 p-4 bg-zinc-900 rounded-xl border border-zinc-800">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={addSegment}
              disabled={isBusy}
              className="flex items-center gap-2 px-4 py-2.5 bg-teal-600 hover:bg-teal-500 disabled:bg-zinc-700 text-white disabled:text-zinc-500 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
            >
              <Plus className="w-5 h-5" />
              Add Segment
            </button>

            <span className="text-zinc-500 text-sm">
              {segments.length} segment{segments.length !== 1 ? "s" : ""}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={exportPodcastJson}
              disabled={isBusy}
              className="flex items-center gap-2 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800/50 text-zinc-300 hover:text-white disabled:text-zinc-600 rounded-lg font-medium transition-colors border border-zinc-700 disabled:cursor-not-allowed"
            >
              <Download className="w-4 h-4" />
              Export JSON
            </button>

            <label
              className={`flex items-center gap-2 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-white rounded-lg font-medium transition-colors cursor-pointer border border-zinc-700 ${isBusy ? "opacity-50 cursor-not-allowed pointer-events-none" : ""}`}
            >
              <Upload className="w-4 h-4" />
              Import
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={importPodcast}
                disabled={isBusy}
                className="hidden"
              />
            </label>
          </div>
        </div>

        {/* Export Progress Banner */}
        {isExporting && (
          <div className="mb-6 p-4 bg-teal-900/30 rounded-xl border border-teal-600/30">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <Loader2 className="w-5 h-5 text-teal-400 animate-spin" />
                <div>
                  <p className="text-white font-medium">Exporting Audio</p>
                  <p className="text-teal-300 text-sm">{exportProgress}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleCancelExport}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Segments */}
        <div className="space-y-4 mb-8">
          {segments.map((segment, index) => (
            <SegmentCard
              key={segment.id}
              segment={segment}
              index={index}
              voices={voices}
              groupedVoices={groupedVoices}
              isLoadingVoices={isLoadingVoices}
              isPlaying={playingSegmentId === segment.id}
              isBusy={isBusy}
              canDelete={segments.length > 1}
              onUpdate={updateSegment}
              onRemove={removeSegment}
              onPlayingChange={(playing) => setPlayingSegmentId(playing ? segment.id : null)}
              registerPlayFunction={registerPlayFunction}
              unregisterPlayFunction={unregisterPlayFunction}
            />
          ))}
        </div>

        {/* Play All Footer */}
        <div className="sticky bottom-4 p-4 bg-zinc-900 rounded-xl border border-zinc-800 shadow-xl">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <Volume2 className="w-5 h-5 text-teal-400" />
              <div>
                <p className="text-white font-medium">Full Podcast</p>
                <p className="text-zinc-500 text-sm">
                  {isPlayingAll
                    ? `Playing segment ${currentPlayingIndex + 1} of ${segments.length}`
                    : `${segments.length} segments ready`}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Download Audio Button */}
              {!isPlayingAll && (
                <button
                  type="button"
                  onClick={handleExportAudio}
                  disabled={segments.every((s) => !s.text.trim()) || isExporting}
                  className="flex items-center gap-2 px-5 py-3 bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 text-white disabled:text-zinc-500 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
                >
                  <FileAudio className="w-5 h-5" />
                  Download Audio
                </button>
              )}

              {/* Play/Stop Button */}
              {isPlayingAll ? (
                <button
                  type="button"
                  onClick={handleStopAll}
                  className="flex items-center gap-2 px-6 py-3 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors"
                >
                  <Square className="w-5 h-5" />
                  Stop Podcast
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handlePlayAll}
                  disabled={segments.every((s) => !s.text.trim()) || isExporting}
                  className="flex items-center gap-2 px-6 py-3 bg-teal-600 hover:bg-teal-500 disabled:bg-zinc-700 text-white disabled:text-zinc-500 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
                >
                  <Play className="w-5 h-5" />
                  Play Podcast
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Individual segment card component
 */
function SegmentCard({
  segment,
  index,
  // voices,
  groupedVoices,
  isLoadingVoices,
  // isPlaying,
  isBusy,
  canDelete,
  onUpdate,
  onRemove,
  onPlayingChange,
  registerPlayFunction,
  unregisterPlayFunction,
}: {
  segment: PodcastSegment;
  index: number;
  voices: string[];
  groupedVoices: Record<string, string[]>;
  isLoadingVoices: boolean;
  isPlaying: boolean;
  isBusy: boolean;
  canDelete: boolean;
  onUpdate: (id: string, field: keyof PodcastSegment, value: string) => void;
  onRemove: (id: string) => void;
  onPlayingChange: (playing: boolean) => void;
  registerPlayFunction: (id: string, playFn: () => Promise<void>) => void;
  unregisterPlayFunction: (id: string) => void;
}) {
  const { read, stop, isReading, isConnecting, error } = useVibeVoice(
    { api: API_BASE },
    { model: "microsoft/VibeVoice-Realtime-0.5B" },
    { speaker_name: segment.voice },
    { device: "cuda" },
  );

  // Create a play function that returns a promise
  const playSegment = useCallback(async () => {
    if (!segment.text.trim()) return;

    try {
      await read(segment.text, {
        cfg_scale: 1.5,
        inference_steps: 5,
      });
    } catch (err) {
      console.error("Failed to read:", err);
      throw err;
    }
  }, [segment.text, read]);

  // Register this segment's play function with the parent
  useEffect(() => {
    registerPlayFunction(segment.id, playSegment);
    return () => {
      unregisterPlayFunction(segment.id);
    };
  }, [segment.id, playSegment, registerPlayFunction, unregisterPlayFunction]);

  const handlePlay = async () => {
    if (!segment.text.trim()) return;
    onPlayingChange(true);
    try {
      await playSegment();
    } catch (err) {
      console.error("Failed to read:", err);
    }
    onPlayingChange(false);
  };

  const handleStop = () => {
    stop();
    onPlayingChange(false);
  };

  const isActive = isReading || isConnecting;

  return (
    <div
      className={`group relative p-5 bg-zinc-900 rounded-xl border transition-colors ${
        isActive ? "border-teal-600/50" : "border-zinc-800 hover:border-zinc-700"
      }`}
    >
      {/* Segment number indicator */}
      <div className="absolute -left-3 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <GripVertical className="w-4 h-4 text-zinc-600" />
      </div>

      <div
        className={`absolute -left-0.5 top-1/2 -translate-y-1/2 w-1 h-12 rounded-full transition-colors ${
          isActive ? "bg-teal-500" : "bg-zinc-700 group-hover:bg-zinc-600"
        }`}
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="flex items-center justify-center w-8 h-8 bg-zinc-800 rounded-lg text-zinc-400 text-sm font-medium">
            {index + 1}
          </span>
          <h3 className="text-white font-medium">Segment {index + 1}</h3>
          {isActive && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 bg-teal-600/20 rounded-full">
              {isConnecting ? (
                <Loader2 className="w-3 h-3 text-teal-400 animate-spin" />
              ) : (
                <span className="w-2 h-2 bg-teal-400 rounded-full animate-pulse" />
              )}
              <span className="text-teal-300 text-xs font-medium">
                {isConnecting ? "Connecting..." : "Playing..."}
              </span>
            </span>
          )}
        </div>

        {canDelete && (
          <button
            type="button"
            onClick={() => onRemove(segment.id)}
            disabled={isActive || isBusy}
            className="p-2 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Voice selector and play button */}
      <div className="flex items-center gap-3 mb-4">
        <div className="flex-1">
          <label className="block text-zinc-400 text-xs font-medium mb-1.5">Voice</label>
          <select
            value={segment.voice}
            onChange={(e) => onUpdate(segment.id, "voice", e.target.value)}
            disabled={isActive || isLoadingVoices || isBusy}
            className="w-full px-3 py-2.5 bg-zinc-800 border border-zinc-700 rounded-lg text-white text-sm focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed appearance-none cursor-pointer"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%239ca3af' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E")`,
              backgroundRepeat: "no-repeat",
              backgroundPosition: "right 0.75rem center",
            }}
          >
            {isLoadingVoices ? (
              <option>Loading voices...</option>
            ) : (
              Object.entries(groupedVoices).map(([lang, langVoices]) => (
                <optgroup key={lang} label={languageNames[lang] || lang.toUpperCase()}>
                  {langVoices.map((voice) => (
                    <option key={voice} value={voice}>
                      {voice.replace(/_/g, " ").replace(/-/g, " - ")}
                    </option>
                  ))}
                </optgroup>
              ))
            )}
          </select>
        </div>

        {isActive ? (
          <button
            type="button"
            onClick={handleStop}
            className="mt-5 flex items-center gap-2 px-4 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors"
          >
            <Square className="w-4 h-4" />
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={handlePlay}
            disabled={!segment.text.trim() || isBusy}
            className="mt-5 flex items-center gap-2 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800/50 text-white disabled:text-zinc-600 rounded-lg font-medium transition-colors border border-zinc-700 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            Play
          </button>
        )}
      </div>

      {/* Text input */}
      <div>
        <label className="block text-zinc-400 text-xs font-medium mb-1.5">Text Content</label>
        <textarea
          value={segment.text}
          onChange={(e) => onUpdate(segment.id, "text", e.target.value)}
          placeholder="Enter text for this segment..."
          disabled={isActive || isBusy}
          rows={3}
          className="w-full px-4 py-3 bg-zinc-800 border border-zinc-700 rounded-lg text-white placeholder-zinc-500 text-sm resize-none focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        />
      </div>

      {/* Error display */}
      {error && !isBusy && (
        <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <p className="text-red-300 text-sm">⚠️ {error}</p>
        </div>
      )}
    </div>
  );
}

export default PodcastMaker;
