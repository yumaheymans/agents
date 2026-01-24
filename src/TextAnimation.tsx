import React from 'react';
import { AbsoluteFill, interpolate, useCurrentFrame, useVideoConfig } from 'remotion';

interface TextAnimationProps {
  text: string;
  animationType: 'glitch' | 'neon' | 'bounce' | 'fade' | '3d' | 'split' | 'wave' | 'rainbow';
  startFrame?: number;
  duration?: number;
}

export const TextAnimation: React.FC<TextAnimationProps> = ({
  text,
  animationType,
  startFrame = 0,
  duration = 60,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const localFrame = frame - startFrame;

  // Only render if within duration
  if (localFrame < 0 || localFrame > duration) {
    return null;
  }

  const progress = interpolate(localFrame, [0, duration], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const renderGlitch = () => {
    const glitchIntensity = Math.sin(frame * 0.5) * 10;
    const redShift = Math.sin(frame * 0.3) * 5;
    const blueShift = Math.sin(frame * 0.4) * 5;

    return (
      <div style={{ position: 'relative' }}>
        <div
          style={{
            fontSize: 120,
            fontWeight: 'bold',
            color: 'cyan',
            position: 'absolute',
            transform: `translate(${redShift}px, ${glitchIntensity}px)`,
            opacity: 0.8,
            mixBlendMode: 'screen',
          }}
        >
          {text}
        </div>
        <div
          style={{
            fontSize: 120,
            fontWeight: 'bold',
            color: 'magenta',
            position: 'absolute',
            transform: `translate(${-blueShift}px, ${-glitchIntensity}px)`,
            opacity: 0.8,
            mixBlendMode: 'screen',
          }}
        >
          {text}
        </div>
        <div
          style={{
            fontSize: 120,
            fontWeight: 'bold',
            color: 'white',
            position: 'relative',
          }}
        >
          {text}
        </div>
      </div>
    );
  };

  const renderNeon = () => {
    const glow = Math.abs(Math.sin(frame * 0.1)) * 30 + 10;

    return (
      <div
        style={{
          fontSize: 120,
          fontWeight: 'bold',
          color: '#ff00ff',
          textShadow: `
            0 0 ${glow}px #ff00ff,
            0 0 ${glow * 2}px #ff00ff,
            0 0 ${glow * 3}px #ff00ff,
            0 0 ${glow * 4}px #ff00ff,
            0 0 ${glow * 5}px #ff00ff
          `,
          animation: 'flicker 0.5s infinite alternate',
        }}
      >
        {text}
      </div>
    );
  };

  const renderBounce = () => {
    const bounce = Math.abs(Math.sin(localFrame * 0.2)) * 100;
    const rotation = Math.sin(localFrame * 0.15) * 15;

    return (
      <div
        style={{
          fontSize: 120,
          fontWeight: 'bold',
          background: 'linear-gradient(45deg, #ff0080, #ff8c00, #40e0d0)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          transform: `translateY(${-bounce}px) rotate(${rotation}deg) scale(${1 + progress * 0.2})`,
        }}
      >
        {text}
      </div>
    );
  };

  const renderFade = () => {
    const scale = interpolate(localFrame, [0, duration / 2, duration], [0, 1.2, 1]);
    const opacity = interpolate(localFrame, [0, 10, duration - 10, duration], [0, 1, 1, 0]);

    return (
      <div
        style={{
          fontSize: 120,
          fontWeight: 'bold',
          background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          opacity,
          transform: `scale(${scale})`,
        }}
      >
        {text}
      </div>
    );
  };

  const render3D = () => {
    const rotateX = interpolate(localFrame, [0, duration], [0, 360]);
    const rotateY = interpolate(localFrame, [0, duration], [0, 720]);

    return (
      <div
        style={{
          fontSize: 120,
          fontWeight: 'bold',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          transform: `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`,
          textShadow: '5px 5px 10px rgba(0,0,0,0.3)',
        }}
      >
        {text}
      </div>
    );
  };

  const renderSplit = () => {
    const letters = text.split('');

    return (
      <div style={{ display: 'flex', gap: '10px' }}>
        {letters.map((letter, index) => {
          const letterDelay = index * 3;
          const letterFrame = Math.max(0, localFrame - letterDelay);
          const letterScale = interpolate(letterFrame, [0, 10], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });
          const letterRotate = interpolate(letterFrame, [0, 20], [180, 0], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          });

          const hue = (index * 30 + frame * 2) % 360;

          return (
            <div
              key={index}
              style={{
                fontSize: 120,
                fontWeight: 'bold',
                color: `hsl(${hue}, 100%, 50%)`,
                transform: `scale(${letterScale}) rotate(${letterRotate}deg)`,
                textShadow: '0 0 20px rgba(255,255,255,0.5)',
              }}
            >
              {letter}
            </div>
          );
        })}
      </div>
    );
  };

  const renderWave = () => {
    const letters = text.split('');

    return (
      <div style={{ display: 'flex' }}>
        {letters.map((letter, index) => {
          const wave = Math.sin((localFrame - index * 2) * 0.3) * 50;
          const hue = (index * 40 + frame) % 360;

          return (
            <div
              key={index}
              style={{
                fontSize: 120,
                fontWeight: 'bold',
                color: `hsl(${hue}, 100%, 60%)`,
                transform: `translateY(${wave}px)`,
                textShadow: `0 0 20px hsl(${hue}, 100%, 60%)`,
              }}
            >
              {letter}
            </div>
          );
        })}
      </div>
    );
  };

  const renderRainbow = () => {
    const hue = (frame * 5) % 360;
    const scale = 1 + Math.sin(localFrame * 0.1) * 0.2;

    return (
      <div
        style={{
          fontSize: 120,
          fontWeight: 'bold',
          background: `linear-gradient(90deg,
            hsl(${hue}, 100%, 50%),
            hsl(${(hue + 60) % 360}, 100%, 50%),
            hsl(${(hue + 120) % 360}, 100%, 50%),
            hsl(${(hue + 180) % 360}, 100%, 50%),
            hsl(${(hue + 240) % 360}, 100%, 50%),
            hsl(${(hue + 300) % 360}, 100%, 50%)
          )`,
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          transform: `scale(${scale})`,
          textShadow: '0 0 30px rgba(255,255,255,0.5)',
        }}
      >
        {text}
      </div>
    );
  };

  const animations = {
    glitch: renderGlitch,
    neon: renderNeon,
    bounce: renderBounce,
    fade: renderFade,
    '3d': render3D,
    split: renderSplit,
    wave: renderWave,
    rainbow: renderRainbow,
  };

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#000',
      }}
    >
      {animations[animationType]()}
    </AbsoluteFill>
  );
};
