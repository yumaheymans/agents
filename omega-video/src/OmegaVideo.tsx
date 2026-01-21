import { AbsoluteFill, Sequence } from 'remotion';
import { OpeningScene } from './scenes/OpeningScene';
import { TaglineScene } from './scenes/TaglineScene';
import { FeaturesScene } from './scenes/FeaturesScene';
import { CTAScene } from './scenes/CTAScene';

export const OmegaVideo: React.FC = () => {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#0a0e27',
      }}
    >
      {/* Opening: Company name reveal (0-90 frames = 0-3s) */}
      <Sequence from={0} durationInFrames={90}>
        <OpeningScene />
      </Sequence>

      {/* Tagline: The Virtual Workforce (90-180 frames = 3-6s) */}
      <Sequence from={90} durationInFrames={90}>
        <TaglineScene />
      </Sequence>

      {/* Features: AI Workers showcase (180-360 frames = 6-12s) */}
      <Sequence from={180} durationInFrames={180}>
        <FeaturesScene />
      </Sequence>

      {/* CTA: Try AI Workers (360-450 frames = 12-15s) */}
      <Sequence from={360} durationInFrames={90}>
        <CTAScene />
      </Sequence>
    </AbsoluteFill>
  );
};
