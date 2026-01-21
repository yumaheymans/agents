import { AbsoluteFill, useCurrentFrame, interpolate, spring, useVideoConfig } from 'remotion';

export const CTAScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Button animation with spring
  const buttonScale = spring({
    frame: frame - 20,
    fps,
    config: {
      damping: 100,
      stiffness: 200,
    },
  });

  // Pulse effect for button
  const pulseScale = 1 + Math.sin(frame / 10) * 0.05;

  // Text fade in
  const textOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const textY = interpolate(frame, [0, 20], [30, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // URL fade in
  const urlOpacity = interpolate(frame, [40, 60], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Glow effect
  const glowIntensity = interpolate(frame, [0, 30], [0, 1], {
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
      {/* Animated background circles */}
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: '800px',
          height: '800px',
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          background: `radial-gradient(circle, #0047FF${Math.floor(glowIntensity * 30).toString(16).padStart(2, '0')} 0%, transparent 70%)`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '50%',
          width: '600px',
          height: '600px',
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          background: `radial-gradient(circle, #00F0FF${Math.floor(glowIntensity * 20).toString(16).padStart(2, '0')} 0%, transparent 70%)`,
        }}
      />

      <div
        style={{
          textAlign: 'center',
          zIndex: 1,
        }}
      >
        {/* Main CTA text */}
        <div
          style={{
            opacity: textOpacity,
            transform: `translateY(${textY}px)`,
            marginBottom: '50px',
          }}
        >
          <div
            style={{
              fontSize: '72px',
              fontWeight: 900,
              fontFamily: 'system-ui, -apple-system, sans-serif',
              color: '#ffffff',
              marginBottom: '20px',
            }}
          >
            Ready to Transform Your Business?
          </div>
          <div
            style={{
              fontSize: '36px',
              fontWeight: 400,
              color: '#00F0FF',
              fontFamily: 'system-ui, -apple-system, sans-serif',
            }}
          >
            Join the future of work today
          </div>
        </div>

        {/* CTA Button */}
        <div
          style={{
            transform: `scale(${buttonScale * pulseScale})`,
            marginBottom: '40px',
          }}
        >
          <div
            style={{
              display: 'inline-block',
              background: 'linear-gradient(135deg, #0047FF 0%, #00F0FF 100%)',
              padding: '25px 80px',
              borderRadius: '50px',
              fontSize: '42px',
              fontWeight: 700,
              color: '#ffffff',
              fontFamily: 'system-ui, -apple-system, sans-serif',
              boxShadow: `0 0 ${glowIntensity * 40}px #0047FF80`,
              cursor: 'pointer',
              transition: 'all 0.3s ease',
            }}
          >
            Try AI Workers
          </div>
        </div>

        {/* Website URL */}
        <div
          style={{
            opacity: urlOpacity,
            fontSize: '32px',
            fontWeight: 600,
            color: '#FAFF00',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            letterSpacing: '0.05em',
          }}
        >
          o-mega.ai
        </div>
      </div>

      {/* Decorative corner accents */}
      {[
        { top: '10%', left: '10%', rotation: 0 },
        { top: '10%', right: '10%', rotation: 90 },
        { bottom: '10%', left: '10%', rotation: -90 },
        { bottom: '10%', right: '10%', rotation: 180 },
      ].map((pos, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            ...pos,
            width: '60px',
            height: '60px',
            border: '3px solid #0047FF',
            borderRight: 'none',
            borderBottom: 'none',
            transform: `rotate(${pos.rotation}deg)`,
            opacity: interpolate(frame, [30 + i * 5, 45 + i * 5], [0, 0.6], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            }),
          }}
        />
      ))}
    </AbsoluteFill>
  );
};
