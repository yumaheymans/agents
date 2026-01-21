import { AbsoluteFill, useCurrentFrame, interpolate } from 'remotion';

export const TaglineScene: React.FC = () => {
  const frame = useCurrentFrame();

  // Typewriter effect for main tagline
  const taglineText = 'The Virtual Workforce';
  const charsToShow = Math.floor(
    interpolate(frame, [0, 40], [0, taglineText.length], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    })
  );

  // Subtitle fade in
  const subtitleOpacity = interpolate(frame, [45, 60], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Particles animation
  const particles = Array.from({ length: 20 }, (_, i) => {
    const angle = (i / 20) * Math.PI * 2;
    const distance = interpolate(frame, [20, 70], [0, 300], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    const opacity = interpolate(frame, [20, 50, 70], [0, 0.6, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });

    return {
      x: Math.cos(angle) * distance,
      y: Math.sin(angle) * distance,
      opacity,
    };
  });

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'center',
        alignItems: 'center',
        background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)',
      }}
    >
      {/* Animated particles */}
      {particles.map((particle, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: '50%',
            top: '50%',
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: i % 3 === 0 ? '#0047FF' : i % 3 === 1 ? '#00F0FF' : '#FAFF00',
            transform: `translate(calc(-50% + ${particle.x}px), calc(-50% + ${particle.y}px))`,
            opacity: particle.opacity,
          }}
        />
      ))}

      {/* Main tagline with typewriter effect */}
      <div
        style={{
          textAlign: 'center',
          maxWidth: '900px',
          padding: '0 40px',
        }}
      >
        <div
          style={{
            fontSize: '80px',
            fontWeight: 900,
            fontFamily: 'system-ui, -apple-system, sans-serif',
            color: '#ffffff',
            lineHeight: 1.2,
            marginBottom: '30px',
            display: 'flex',
            justifyContent: 'center',
            position: 'relative',
          }}
        >
          <span>
            {taglineText.slice(0, charsToShow)}
            {charsToShow < taglineText.length && frame < 45 && (
              <span
                style={{
                  borderRight: '4px solid #00F0FF',
                  animation: 'blink 0.7s infinite',
                }}
              >
                &nbsp;
              </span>
            )}
          </span>
        </div>

        {/* Subtitle */}
        <div
          style={{
            fontSize: '36px',
            fontWeight: 400,
            color: '#00F0FF',
            opacity: subtitleOpacity,
            fontFamily: 'system-ui, -apple-system, sans-serif',
          }}
        >
          Deploy autonomous AI agents to scale your business
        </div>
      </div>

      {/* Accent glow effect */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: '600px',
          height: '600px',
          transform: 'translate(-50%, -50%)',
          background: 'radial-gradient(circle, #0047FF20 0%, transparent 70%)',
          opacity: interpolate(frame, [0, 30], [0, 1], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
        }}
      />
    </AbsoluteFill>
  );
};
