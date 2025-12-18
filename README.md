# 🎙️ VibeVoice Podcast Generator

Create professional multi-speaker podcasts with AI text-to-speech. Manage speakers, assign voices, and generate high-quality audio with an intuitive interface.

> **Note**: This project is a fork of [vibevoice-podcast](https://github.com/skorotkiewicz/vibevoice-podcast) by skorotkiewicz, enhanced with multi-speaker support, sidebar layout, and theme switching.

## ✨ Features

### Speaker Management
- **Multi-speaker support** — Add unlimited speakers with unique voices
- **Speaker library** — Manage all speakers in a dedicated sidebar
- **Color-coded speakers** — Visual identification with unique colors (8 unique colors cycling)
- **Voice assignment** — Choose from 25+ voices per speaker
- **Easy editing** — Rename speakers and change voices on the fly

### Content Creation
- **Segment-based editing** — Add unlimited text segments
- **Speaker assignment** — Assign any speaker to any segment
- **Real-time preview** — Play individual segments or the entire podcast
- **Audio caching** — Generated clips cached for instant replay
- **Smart generation** — Generate all missing audio with one click

### User Experience
- **Sidebar layout** — Fixed sidebar with scrollable content area
- **Light/Dark mode** — Toggle between themes with one click
- **Responsive design** — Works on desktop, tablet, and mobile
- **Modern UI** — Clean, professional interface with smooth animations

### Export & Import
- **Audio export** — Download complete podcast as WAV file
- **Project management** — Save/load projects as JSON
- **Batch operations** — Generate all segments at once

---

## 🎬 Demo

<img width="2531" height="1280" alt="Podcast-Generator-12-18-2025_06_43_AM" src="https://github.com/user-attachments/assets/b0c035e8-991f-4e7b-b8b1-1c4329985538" />


### 🎧 Example Podcast

<audio src="example/podcast-1766008504673.wav" controls>
  Your browser does not support the audio element. <a href="example/podcast-1766008504673.wav">Download the example podcast</a>
</audio>

> **Note**: If the player doesn't appear, you can [download the WAV file](example/podcast-1766008504673.wav) to listen.


---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+ (or Bun)
- NVIDIA GPU with CUDA support (recommended)
- Git

### 1. Clone This Repository
```bash
# Clone VibeVoice Poidcast Generator
git clone https://github.com/wpacademy/vibevoice-podcast-generator.git
cd vibevoice-podcast-generator
```
### 2. Install and Configure VibeVoice

Follow the comprehensive [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions on:
- Cloning the VibeVoice repository
- Setting up Python virtual environment
- Installing dependencies with CUDA support
- Downloading voice presets
- Starting the API server


> **Important**: Use port `8880` for the server to match the frontend configuration.

### 3. Run the Frontend

```bash
# Install dependencies
npm install   # or: bun install

# Start development server
npm run dev   # or: bun dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 📖 Usage Guide

### Managing Speakers

1. **Add Speaker** — Click the **+** button in the sidebar
2. **Name Speaker** — Click the speaker name to edit (e.g., "Host", "Guest")
3. **Select Voice** — Choose from 25+ voices in multiple languages
4. **Delete Speaker** — Click the trash icon (minimum 1 speaker required)

Each speaker gets a unique color for easy identification! Colors cycle through 8 distinct options.

### Creating Content

1. **Add Segment** — Click "Add Segment" button
2. **Write Text** — Enter dialogue or narration
3. **Assign Speaker** — Select speaker from dropdown
4. **Generate Audio** — Click lightning icon to create audio

### Multi-Speaker Conversations

Create natural conversations by alternating speakers:

```
Segment 1: [Host] "Welcome to our podcast!"
Segment 2: [Guest] "Thanks for having me!"
Segment 3: [Host] "Let's dive into today's topic."
```

### Audio Generation

- **Generate** (⚡) — Creates audio for a segment
- **Generate All** (🔄) — Generates all missing audio
- **Ready** (✓) — Green indicator shows cached audio
- **Regenerate** — Click again to create a new version

> **Tip**: Audio is cached! Once generated, it plays instantly every time.

### Playback

- **Play Segment** — Click play button on individual segments
- **Play All** — Plays entire podcast with all speakers
- **Stop** — Stop playback at any time

### Theme Toggle

Click the **Sun/Moon** icon in the sidebar to switch between:
- 🌙 **Dark Mode** — Easy on the eyes (default)
- ☀️ **Light Mode** — Bright and clear

### Exporting

**Download Audio**
- Click "Download Audio" to export complete podcast
- Automatically generates missing segments
- Exports as high-quality WAV file

**Export JSON**
- Save your project for later editing
- Preserves all speakers and segments
- Import anytime to continue work

---

## 🎨 Available Voices

### English
- Emma (woman), Carter (man), Sophia (woman), Liam (man)
- Olivia (woman), Noah (man), Ava (woman), Ethan (man)
- Isabella (woman), Mason (man), Mia (woman)

### Other Languages
- 🇩🇪 German, 🇫🇷 French, 🇮🇹 Italian, 🇪🇸 Spanish
- 🇵🇹 Portuguese, 🇳🇱 Dutch, 🇵🇱 Polish
- 🇯🇵 Japanese, 🇰🇷 Korean, 🇮🇳 Indian English

---

## ⚙️ Configuration

### API Endpoint

The frontend connects to `http://localhost:8880/api`. To change this, edit `API_BASE` in `src/App.tsx`:

```typescript
const API_BASE = "http://localhost:8880/api";
```

### Server Options

```bash
python demo/server.py \
  --model microsoft/VibeVoice-Realtime-0.5B \
  --device cuda \
  --port 8880 \
  --inference_steps 5
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed server configuration options.

---

## 🏗️ Tech Stack

**Frontend**
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Lucide React (icons)

**Backend**
- FastAPI (API server)
- VibeVoice TTS (Microsoft)
- PyTorch (inference)

**Features**
- Client-side audio caching
- Real-time synthesis
- Responsive design
- Theme switching

---

## 📁 Project Structure

```
vibevoice-podcast-generator/
├── src/
│   ├── App.tsx              # Main application
│   ├── components/
│   │   └── Sidebar.tsx      # Speaker management sidebar
│   └── types/
│       └── models.ts        # TypeScript interfaces
├── VibeVoice/
│   └── demo/
│       └── server.py        # API server
├── example/                 # Sample podcasts
├── SETUP_GUIDE.md          # Detailed setup instructions
└── README.md
```

---

## 🎯 Tips & Best Practices

1. **Pre-generate audio** — Generate all segments before playback for smooth experience
2. **Use descriptive names** — Name speakers clearly (e.g., "Host", "Expert", "Narrator")
3. **Match voices to roles** — Choose appropriate voices for each character
4. **Test segments** — Play individual segments to verify before exporting
5. **Save often** — Export JSON regularly to preserve your work
6. **Use theme toggle** — Switch to light mode for better visibility in bright environments

---

## 🐛 Troubleshooting

**Server won't start**
- Check if port 8880 is available
- Verify CUDA is installed (for GPU)
- Try `--device cpu` for CPU-only mode
- See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting

**Audio not generating**
- Check server logs for errors
- Verify text is not empty
- Ensure speaker has a voice assigned
- Check that server is running on port 8880

**CORS errors**
- Ensure server is running on port 8880
- Check browser console for details
- Verify API_BASE matches server port

**Theme not switching**
- Refresh the page
- Check browser console for errors
- Clear browser cache if needed

For more troubleshooting help, see [SETUP_GUIDE.md](SETUP_GUIDE.md).

---

## 📝 Example Workflows

### Interview Podcast

1. Add two speakers: "Host" and "Guest"
2. Assign different voices (e.g., Emma and Carter)
3. Alternate segments between speakers
4. Generate all and export

### Narrated Story

1. Add "Narrator" speaker
2. Create segments for each paragraph
3. Add optional "Character" speakers for dialogue
4. Generate and download

### Educational Content

1. Add "Teacher" and "Student" speakers
2. Create Q&A format segments
3. Use different voices for clarity
4. Export for distribution

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

---

## 📄 License

MIT License - feel free to use this project for any purpose.

---

## 🙏 Acknowledgments

- **[skorotkiewicz/vibevoice-podcast](https://github.com/skorotkiewicz/vibevoice-podcast)** — Original fork source and inspiration
- **Microsoft VibeVoice** — Excellent TTS model and API
- **React Team** — Amazing framework
- **Tailwind CSS** — Beautiful styling system
- **Lucide Icons** — Clean, modern icon set

---

## 🔗 Links

- [Original Fork](https://github.com/skorotkiewicz/vibevoice-podcast)
- [VibeVoice GitHub](https://github.com/microsoft/VibeVoice)
- [VibeVoice Model (HuggingFace)](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- [Setup Guide](SETUP_GUIDE.md)

---

**Made with ❤️ using VibeVoice TTS**
