import React from 'react';
import { Sequence } from 'remotion';
import { TextAnimation } from './TextAnimation';

export const FlashyTextComposition: React.FC = () => {
  const animationDuration = 90; // 3 seconds at 30fps
  const texts = [
    { text: 'GLITCH', type: 'glitch' as const },
    { text: 'NEON GLOW', type: 'neon' as const },
    { text: 'BOUNCE!', type: 'bounce' as const },
    { text: 'FADE IN', type: 'fade' as const },
    { text: 'ROTATE 3D', type: '3d' as const },
    { text: 'SPLIT TEXT', type: 'split' as const },
    { text: 'WAVE', type: 'wave' as const },
    { text: 'RAINBOW', type: 'rainbow' as const },
  ];

  return (
    <>
      {texts.map((item, index) => (
        <Sequence
          key={index}
          from={index * animationDuration}
          durationInFrames={animationDuration}
        >
          <TextAnimation
            text={item.text}
            animationType={item.type}
            duration={animationDuration}
          />
        </Sequence>
      ))}
    </>
  );
};
