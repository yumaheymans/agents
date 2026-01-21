import { AbsoluteFill, useCurrentFrame, useVideoConfig, interpolate, spring } from 'remotion';

export const OpeningScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Logo scale animation with spring
  const scale = spring({
    frame: frame - 10,
    fps,
    config: {
      damping: 100,
      stiffness: 200,
      mass: 0.5,
    },
  });

  // Text fade and slide in
  const textOpacity = interpolate(frame, [30, 50], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const textY = interpolate(frame, [30, 50], [30, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Accent lines animation
  const line1Width = interpolate(frame, [20, 40], [0, 100], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const line2Width = interpolate(frame, [25, 45], [0, 100], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)',
      }}
    >
      {/* Background grid effect */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: `linear-gradient(#0047FF15 1px, transparent 1px),
                            linear-gradient(90deg, #0047FF15 1px, transparent 1px)`,
          backgroundSize: '50px 50px',
          opacity: 0.3,
        }}
      />

      {/* Accent lines */}
      <div
        style={{
          position: 'absolute',
          top: '45%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '600px',
          height: '2px',
        }}
      >
        <div
          style={{
            width: `${line1Width}%`,
            height: '2px',
            background: 'linear-gradient(90deg, transparent, #00F0FF, transparent)',
            marginBottom: '8px',
          }}
        />
        <div
          style={{
            width: `${line2Width}%`,
            height: '2px',
            background: 'linear-gradient(90deg, transparent, #FAFF00, transparent)',
          }}
        />
      </div>

      {/* Company logo/name */}
      <div
        style={{
          transform: `scale(${scale})`,
          marginBottom: '20px',
        }}
      >
        <div
          style={{
            fontSize: '120px',
            fontWeight: 900,
            fontFamily: 'system-ui, -apple-system, sans-serif',
            background: 'linear-gradient(135deg, #0047FF 0%, #00F0FF 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            letterSpacing: '-0.02em',
            textTransform: 'lowercase',
          }}
        >
          o-mega
        </div>
      </div>

      {/* Subtitle */}
      <div
        style={{
          opacity: textOpacity,
          transform: `translateY(${textY}px)`,
          fontSize: '32px',
          fontWeight: 600,
          color: '#FAFF00',
          fontFamily: 'system-ui, -apple-system, sans-serif',
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
        }}
      >
        AI-Powered Innovation
      </div>
    </AbsoluteFill>
  );
};
