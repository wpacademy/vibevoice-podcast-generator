import {
  Check,
  Download,
  FileAudio,
  GripVertical,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Square,
  Trash2,
  Upload,
  Volume2,
} from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ConfigResponse, ModelType } from "./types/models";
import { detectModelType } from "./types/models";
import { Sidebar } from "./components/Sidebar";

const API_BASE = "http://localhost:8880/api";
const SAMPLE_RATE = 24_000;

interface Speaker {
  id: string;
  name: string;
  voice: string;
  color: string;
}

interface PodcastSegment {
  id: string;
  text: string;
  voice: string;
  speakerId?: string; // NEW: Reference to speaker
}

interface PodcastData {
  segments: PodcastSegment[];
  createdAt: string;
  version: string;
}

// Cached audio for each segment (stored as ArrayBuffer for WAV data)
interface SegmentAudioCache {
  [segmentId: string]: {
    audioData: ArrayBuffer;
    text: string;
    voice: string;
  };
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
 * Generate audio for a single segment via /api/synthesize
 * Returns WAV audio data as ArrayBuffer
 */
async function synthesizeSegmentAudio(
  text: string,
  voice: string,
  onProgress?: (message: string) => void,
): Promise<ArrayBuffer> {
  onProgress?.("Generating...");

  const response = await fetch(`${API_BASE}/synthesize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      voice,
      cfg_scale: 1.5,
      inference_steps: 5,
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to synthesize audio: ${response.statusText}`);
  }

  onProgress?.("Done!");
  return await response.arrayBuffer();
}

/**
 * Extract PCM16 data from WAV ArrayBuffer (skip 44-byte header)
 */
function wavToPcm16(wavData: ArrayBuffer): Int16Array {
  // WAV header is 44 bytes, PCM16 data follows
  return new Int16Array(wavData, 44);
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
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
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
 * This version uses /api/synthesize for pre-generating and caching audio
 */
export function PodcastMaker() {
  const [segments, setSegments] = useState<PodcastSegment[]>([
    {
      id: crypto.randomUUID(),
      text: "Welcome to our podcast! Today we'll be discussing exciting topics.",
      voice: "en-Emma_woman",
      speakerId: "1", // Default to first speaker
    },
  ]);

  const [voices, setVoices] = useState<string[]>([]);
  const [isLoadingVoices, setIsLoadingVoices] = useState(true);
  const [modelType, setModelType] = useState<ModelType | null>(null);

  // Speaker management
  const [speakers, setSpeakers] = useState<Speaker[]>([
    {
      id: "1",
      name: "Host",
      voice: "en-Emma_woman",
      color: "#3b82f6", // blue-500
    },
  ]);

  // Theme state
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');
  const [playingSegmentId, setPlayingSegmentId] = useState<string | null>(null);
  const [isPlayingAll, setIsPlayingAll] = useState(false);
  const [currentPlayingIndex, setCurrentPlayingIndex] = useState<number>(-1);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<string>("");

  // Audio cache - stores generated audio for each segment
  const [audioCache, setAudioCache] = useState<SegmentAudioCache>({});
  const [generatingSegmentId, setGeneratingSegmentId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const stopPlayAllRef = useRef(false);
  const stopExportRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);

  // Get or create AudioContext
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === "closed") {
      audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
    }
    if (audioContextRef.current.state === "suspended") {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  // Fetch available voices on mount
  useEffect(() => {
    const fetchVoices = async () => {
      try {
        const response = await fetch(`${API_BASE}/config`);
        if (response.ok) {
          const config: ConfigResponse = await response.json();

          // Detect model type
          const detectedModel = detectModelType(config);
          setModelType(detectedModel);

          // Set voices (both models now return voices array)
          if (config.voices && config.voices.length > 0) {
            setVoices(config.voices);
          } else {
            // Fallback voices if none provided
            setVoices([
              "en-Carter_man",
              "en-Davis_man",
              "en-Emma_woman",
              "en-Frank_man",
              "en-Grace_woman",
              "en-Mike_man",
            ]);
          }

          console.log(`Detected model: ${detectedModel}`, config);
        }
      } catch (err) {
        console.error("Failed to fetch voices:", err);
        setVoices([
          "en-Carter_man",
          "en-Davis_man",
          "en-Emma_woman",
          "en-Frank_man",
          "en-Grace_woman",
          "en-Mike_man",
        ]);
        setModelType('0.5B'); // Assume 0.5B on error
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

  // Helper function to generate random colors for speakers
  // Helper function to get next color for speakers (cycles through available colors)
  const getNextColor = () => {
    const colors = [
      "#3b82f6", // blue
      "#10b981", // green
      "#f59e0b", // amber
      "#ef4444", // red
      "#8b5cf6", // violet
      "#ec4899", // pink
      "#14b8a6", // teal
      "#f97316", // orange
    ];
    // Use speaker count to cycle through colors
    return colors[speakers.length % colors.length];
  };

  // Speaker management functions
  const handleAddSpeaker = () => {
    const newSpeaker: Speaker = {
      id: Date.now().toString(),
      name: `Speaker ${speakers.length + 1}`,
      voice: voices[0] || "en-Emma_woman",
      color: getNextColor(),
    };
    setSpeakers([...speakers, newSpeaker]);
  };

  const handleUpdateSpeaker = (id: string, updates: Partial<Speaker>) => {
    setSpeakers(speakers.map((s) => (s.id === id ? { ...s, ...updates } : s)));
  };

  const handleDeleteSpeaker = (id: string) => {
    if (speakers.length > 1) {
      setSpeakers(speakers.filter((s) => s.id !== id));
      // Update segments that used this speaker
      setSegments(
        segments.map((seg) =>
          seg.speakerId === id ? { ...seg, speakerId: undefined } : seg
        )
      );
    }
  };

  const addSegment = () => {
    const newSegment: PodcastSegment = {
      id: crypto.randomUUID(),
      text: "",
      voice: voices[0] || "en-Emma_woman",
      speakerId: speakers[0]?.id, // Assign first speaker by default
    };
    setSegments([...segments, newSegment]);
  };

  const removeSegment = (id: string) => {
    if (segments.length > 1) {
      setSegments(segments.filter((s) => s.id !== id));
      // Remove from cache
      setAudioCache((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
  };

  const updateSegment = (id: string, field: keyof PodcastSegment, value: string) => {
    setSegments(segments.map((s) => (s.id === id ? { ...s, [field]: value } : s)));
    // Invalidate cache when text or voice changes
    if (field === "text" || field === "voice") {
      setAudioCache((prev) => {
        const next = { ...prev };
        delete next[id];
        return next;
      });
    }
  };

  // Check if segment audio is cached and valid
  const isSegmentCached = (segment: PodcastSegment): boolean => {
    const cached = audioCache[segment.id];
    if (!cached) return false;

    // Get the actual voice that would be used (from speaker if assigned)
    const speaker = speakers.find((s) => s.id === segment.speakerId);
    const actualVoice = speaker ? speaker.voice : segment.voice;

    return cached.text === segment.text && cached.voice === actualVoice;
  };

  // Generate audio for a single segment
  const generateSegmentAudio = async (segment: PodcastSegment) => {
    if (!segment.text.trim()) return;

    // Get voice from speaker if assigned, otherwise use segment voice
    const speaker = speakers.find((s) => s.id === segment.speakerId);
    const voice = speaker ? speaker.voice : segment.voice;

    setGeneratingSegmentId(segment.id);
    try {
      const audioData = await synthesizeSegmentAudio(segment.text, voice);
      setAudioCache((prev) => ({
        ...prev,
        [segment.id]: {
          audioData,
          text: segment.text,
          voice,
        },
      }));
    } catch (err) {
      console.error("Failed to generate audio:", err);
      alert(`Failed to generate audio: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setGeneratingSegmentId(null);
    }
  };

  // Generate all missing audio
  const generateAllAudio = async () => {
    const missingSegments = segments.filter((s) => s.text.trim() && !isSegmentCached(s));

    for (const segment of missingSegments) {
      if (stopExportRef.current) break;
      await generateSegmentAudio(segment);
      await delay(500); // Give server some breathing room
    }
  };

  // Play audio from cache
  const playAudioFromCache = async (segmentId: string): Promise<void> => {
    const cached = audioCache[segmentId];
    if (!cached) return;

    const ctx = getAudioContext();
    const pcmData = wavToPcm16(cached.audioData);

    // Convert PCM16 to Float32
    const float32Array = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      float32Array[i] = pcmData[i] / 32768.0;
    }

    const audioBuffer = ctx.createBuffer(1, float32Array.length, SAMPLE_RATE);
    audioBuffer.getChannelData(0).set(float32Array);

    return new Promise((resolve) => {
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      currentSourceRef.current = source;

      source.onended = () => {
        currentSourceRef.current = null;
        resolve();
      };

      source.start();
    });
  };

  // Stop current playback
  const stopPlayback = () => {
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
        currentSourceRef.current.disconnect();
      } catch { }
      currentSourceRef.current = null;
    }
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
          setSegments(
            data.segments.map((s) => ({
              ...s,
              id: crypto.randomUUID(),
            })),
          );
          // Clear audio cache for new import
          setAudioCache({});
        }
      } catch (err) {
        console.error("Failed to parse podcast file:", err);
        alert("Invalid podcast file format");
      }
    };
    reader.readAsText(file);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Play all segments sequentially
  const handlePlayAll = async () => {
    setIsPlayingAll(true);
    stopPlayAllRef.current = false;

    for (let i = 0; i < segments.length; i++) {
      if (stopPlayAllRef.current) break;

      const segment = segments[i];
      if (!segment.text.trim()) continue;

      setCurrentPlayingIndex(i);
      setPlayingSegmentId(segment.id);

      try {
        // Generate if not cached
        if (!isSegmentCached(segment)) {
          await generateSegmentAudio(segment);
        }

        // Play from cache
        if (audioCache[segment.id] || isSegmentCached(segment)) {
          await playAudioFromCache(segment.id);
        }
      } catch (err) {
        console.error(`Failed to play segment ${i + 1}:`, err);
      }

      if (i < segments.length - 1 && !stopPlayAllRef.current) {
        await delay(300);
      }
    }

    setIsPlayingAll(false);
    setCurrentPlayingIndex(-1);
    setPlayingSegmentId(null);
  };

  const handleStopAll = () => {
    stopPlayAllRef.current = true;
    stopPlayback();
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
    const silenceBetweenSegments = createSilence(100);

    try {
      for (let i = 0; i < validSegments.length; i++) {
        if (stopExportRef.current) {
          setIsExporting(false);
          setExportProgress("");
          return;
        }

        const segment = validSegments[i];

        // Use cached audio if available, otherwise generate
        let wavData: ArrayBuffer;
        if (isSegmentCached(segment)) {
          setExportProgress(`Using cached audio for segment ${i + 1}/${validSegments.length}`);
          wavData = audioCache[segment.id].audioData;
        } else {
          setExportProgress(`Generating segment ${i + 1}/${validSegments.length}...`);
          wavData = await synthesizeSegmentAudio(segment.text, segment.voice, (msg) =>
            setExportProgress(`Segment ${i + 1}/${validSegments.length}: ${msg}`),
          );
          // Cache the newly generated audio
          setAudioCache((prev) => ({
            ...prev,
            [segment.id]: {
              audioData: wavData,
              text: segment.text,
              voice: segment.voice,
            },
          }));
          await delay(500);
        }

        const pcmData = wavToPcm16(wavData);
        audioChunks.push(pcmData);

        if (i < validSegments.length - 1) {
          audioChunks.push(silenceBetweenSegments);
        }
      }

      setExportProgress("Creating audio file...");

      const totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const combined = new Int16Array(totalLength);
      let offset = 0;

      for (const chunk of audioChunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }

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

  // Count cached segments
  const cachedCount = segments.filter((s) => isSegmentCached(s)).length;
  const validCount = segments.filter((s) => s.text.trim()).length;

  const isBusy = isPlayingAll || isExporting || generatingSegmentId !== null;

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  const isDark = theme === 'dark';

  return (
    <div className={`flex min-h-screen ${isDark ? 'bg-zinc-950' : 'bg-gray-50'}`}>
      {/* Sidebar */}
      <Sidebar
        speakers={speakers}
        voices={voices}
        onAddSpeaker={handleAddSpeaker}
        onUpdateSpeaker={handleUpdateSpeaker}
        onDeleteSpeaker={handleDeleteSpeaker}
        modelType={modelType}
        theme={theme}
        onThemeToggle={toggleTheme}
      />

      {/* Main Content - with left margin for fixed sidebar */}
      <main className="flex-1 ml-80 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-8">

          {/* Action Bar */}
          <div className={`flex flex-wrap items-center justify-between gap-4 mb-6 p-4 rounded-xl border ${isDark
            ? 'bg-zinc-900 border-zinc-800'
            : 'bg-white border-gray-200'
            }`}>
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

              <span className={`text-sm ${isDark ? 'text-zinc-500' : 'text-gray-500'}`}>
                {segments.length} segment{segments.length !== 1 ? "s" : ""}
                {validCount > 0 && (
                  <span className="ml-2 text-teal-400">
                    ({cachedCount}/{validCount} generated)
                  </span>
                )}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={generateAllAudio}
                disabled={isBusy || cachedCount === validCount}
                className="flex items-center gap-2 px-4 py-2.5 bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-700 text-white disabled:text-zinc-500 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
              >
                <RefreshCw className="w-4 h-4" />
                Generate All
              </button>

              <button
                type="button"
                onClick={exportPodcastJson}
                disabled={isBusy}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-colors border disabled:cursor-not-allowed ${isDark
                  ? 'bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800/50 text-zinc-300 hover:text-white disabled:text-zinc-600 border-zinc-700'
                  : 'bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 text-gray-700 hover:text-gray-900 disabled:text-gray-400 border-gray-300'
                  }`}
              >
                <Download className="w-4 h-4" />
                Export JSON
              </button>

              <label
                className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-colors cursor-pointer border ${isDark
                  ? 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-white border-zinc-700'
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700 hover:text-gray-900 border-gray-300'
                  } ${isBusy ? "opacity-50 cursor-not-allowed pointer-events-none" : ""}`}
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
                speakers={speakers}
                isBusy={isBusy}
                isPlaying={playingSegmentId === segment.id}
                isGenerating={generatingSegmentId === segment.id}
                isCached={isSegmentCached(segment)}
                canDelete={segments.length > 1}
                theme={theme}
                onUpdate={updateSegment}
                onRemove={removeSegment}
                onGenerate={() => generateSegmentAudio(segment)}
                onPlay={async () => {
                  if (!segment.text.trim()) return;
                  setPlayingSegmentId(segment.id);
                  try {
                    if (!isSegmentCached(segment)) {
                      await generateSegmentAudio(segment);
                    }
                    // Need to use updated cache
                    await playAudioFromCache(segment.id);
                  } catch (err) {
                    console.error("Failed to play:", err);
                  }
                  setPlayingSegmentId(null);
                }}
                onStop={() => {
                  stopPlayback();
                  setPlayingSegmentId(null);
                }}
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
      </main>
    </div>
  );
}

/**
 * Individual segment card component
 */
function SegmentCard({
  segment,
  index,
  speakers,
  isBusy,
  isPlaying,
  isGenerating,
  isCached,
  canDelete,
  theme,
  onUpdate,
  onRemove,
  onGenerate,
  onPlay,
  onStop,
}: {
  segment: PodcastSegment;
  index: number;
  speakers: Speaker[];
  isBusy: boolean;
  isPlaying: boolean;
  isGenerating: boolean;
  isCached: boolean;
  canDelete: boolean;
  theme: 'light' | 'dark';
  onUpdate: (id: string, field: keyof PodcastSegment, value: string) => void;
  onRemove: (id: string) => void;
  onGenerate: () => void;
  onPlay: () => void;
  onStop: () => void;
}) {
  const isActive = isPlaying || isGenerating;
  const isDark = theme === 'dark';

  return (
    <div
      className={`group relative p-5 rounded-xl border transition-colors ${isDark ? 'bg-zinc-900' : 'bg-white'
        } ${isActive ? "border-teal-600/50" : isDark ? "border-zinc-800 hover:border-zinc-700" : "border-gray-200 hover:border-gray-300"
        }`}
    >
      {/* Segment number indicator */}
      <div className="absolute -left-3 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <GripVertical className={`w-4 h-4 ${isDark ? 'text-zinc-600' : 'text-gray-400'}`} />
      </div>

      <div
        className={`absolute -left-0.5 top-1/2 -translate-y-1/2 w-1 h-12 rounded-full transition-colors ${isActive
          ? "bg-teal-500"
          : isCached
            ? "bg-green-500"
            : isDark
              ? "bg-zinc-700 group-hover:bg-zinc-600"
              : "bg-gray-300 group-hover:bg-gray-400"
          }`}
      />

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className={`flex items-center justify-center w-8 h-8 rounded-lg text-sm font-medium ${isDark ? 'bg-zinc-800 text-zinc-400' : 'bg-gray-100 text-gray-600'
            }`}>
            {index + 1}
          </span>
          <h3 className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>Segment {index + 1}</h3>

          {/* Status indicators */}
          {isGenerating && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 bg-amber-600/20 rounded-full">
              <Loader2 className="w-3 h-3 text-amber-400 animate-spin" />
              <span className="text-amber-300 text-xs font-medium">Generating...</span>
            </span>
          )}
          {isPlaying && !isGenerating && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 bg-teal-600/20 rounded-full">
              <span className="w-2 h-2 bg-teal-400 rounded-full animate-pulse" />
              <span className="text-teal-300 text-xs font-medium">Playing...</span>
            </span>
          )}
          {isCached && !isActive && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 bg-green-600/20 rounded-full">
              <Check className="w-3 h-3 text-green-400" />
              <span className="text-green-300 text-xs font-medium">Ready</span>
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

      {/* Speaker and Voice selector */}
      <div className="flex items-center gap-3 mb-4">
        <div className="flex-1">
          <label className={`block text-xs font-medium mb-1.5 ${isDark ? 'text-zinc-400' : 'text-gray-600'}`}>Speaker</label>
          <select
            value={segment.speakerId || ""}
            onChange={(e) => onUpdate(segment.id, "speakerId", e.target.value)}
            disabled={isActive || isBusy}
            className={`w-full px-3 py-2.5 rounded-lg text-sm border focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed appearance-none cursor-pointer ${isDark
              ? 'bg-zinc-800 border-zinc-700 text-white'
              : 'bg-white border-gray-300 text-gray-900'
              }`}
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%239ca3af' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E")`,
              backgroundRepeat: "no-repeat",
              backgroundPosition: "right 0.75rem center",
            }}
          >
            <option value="">Select Speaker</option>
            {speakers.map((speaker) => (
              <option key={speaker.id} value={speaker.id}>
                {speaker.name} ({speaker.voice})
              </option>
            ))}
          </select>
        </div>

        {/* Generate button */}
        <button
          type="button"
          onClick={onGenerate}
          disabled={!segment.text.trim() || isBusy}
          className={`mt-5 flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-colors disabled:cursor-not-allowed ${isCached
            ? "bg-zinc-800 hover:bg-zinc-700 text-zinc-300 border border-zinc-700"
            : "bg-amber-600 hover:bg-amber-500 disabled:bg-zinc-700 text-white disabled:text-zinc-500"
            }`}
        >
          {isGenerating ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4" />
          )}
          {isCached ? "Regenerate" : "Generate"}
        </button>

        {/* Play/Stop button */}
        {isPlaying ? (
          <button
            type="button"
            onClick={onStop}
            className="mt-5 flex items-center gap-2 px-4 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium transition-colors"
          >
            <Square className="w-4 h-4" />
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={onPlay}
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
        <label className={`block text-xs font-medium mb-1.5 ${isDark ? 'text-zinc-400' : 'text-gray-600'}`}>Text Content</label>
        <textarea
          value={segment.text}
          onChange={(e) => onUpdate(segment.id, "text", e.target.value)}
          placeholder="Enter text for this segment..."
          disabled={isActive || isBusy}
          rows={3}
          className={`w-full px-4 py-3 rounded-lg text-sm border resize-none focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${isDark
            ? 'bg-zinc-800 border-zinc-700 text-white placeholder-zinc-500'
            : 'bg-white border-gray-300 text-gray-900 placeholder-gray-400'
            }`}
        />
      </div>
    </div>
  );
}

export default PodcastMaker;
