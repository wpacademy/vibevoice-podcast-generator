/**
 * Type definitions for dual model support (0.5B and 1.5B)
 */

export type ModelType = '0.5B' | '1.5B' | 'auto';

export interface BaseConfigResponse {
  model: string;
  device: string;
  sample_rate: number;
}

export interface Config05B extends BaseConfigResponse {
  voices: string[];
  default_voice: string;
}

export interface Config15B extends BaseConfigResponse {
  voices: string[];  // Fake voices for compatibility
  default_voice: string;
  features: {
    multi_speaker: boolean;
    voice_cloning: boolean;
    max_speakers: number;
  };
}

export type ConfigResponse = Config05B | Config15B;

export interface SynthesizeRequest {
  text: string;
  voice?: string;  // For 0.5B compatibility
  speaker_prompt?: string;  // For 1.5B features
  cfg_scale?: number;
  inference_steps?: number;
  temperature?: number;
  top_p?: number;
}

export interface MultiSpeakerRequest {
  dialogue: Array<{
    speaker: string;
    text: string;
  }>;
  cfg_scale?: number;
  inference_steps?: number;
}

/**
 * Helper function to detect model type from config response
 */
export function detectModelType(config: ConfigResponse): ModelType {
  // Check if config has features (1.5B model)
  if ('features' in config && config.features?.voice_cloning) {
    return '1.5B';
  }
  // Otherwise it's 0.5B
  return '0.5B';
}

/**
 * Check if config is from 0.5B model
 */
export function isConfig05B(config: ConfigResponse): config is Config05B {
  return !('features' in config) || !config.features?.voice_cloning;
}

/**
 * Check if config is from 1.5B model
 */
export function isConfig15B(config: ConfigResponse): config is Config15B {
  return 'features' in config && config.features?.voice_cloning === true;
}
