import { AbsoluteFill, useCurrentFrame, interpolate, Sequence } from 'remotion';

const Feature: React.FC<{
  title: string;
  description: string;
  icon: string;
  delay: number;
}> = ({ title, description, icon, delay }) => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [delay, delay + 15], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const y = interpolate(frame, [delay, delay + 15], [50, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const scale = interpolate(frame, [delay, delay + 15], [0.8, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <div
      style={{
        opacity,
        transform: `translateY(${y}px) scale(${scale})`,
        background: 'linear-gradient(135deg, #0047FF15 0%, #00F0FF10 100%)',
        border: '2px solid #0047FF40',
        borderRadius: '20px',
        padding: '40px',
        minWidth: '350px',
        backdropFilter: 'blur(10px)',
      }}
    >
      <div
        style={{
          fontSize: '48px',
          marginBottom: '20px',
        }}
      >
        {icon}
      </div>
      <div
        style={{
          fontSize: '32px',
          fontWeight: 700,
          color: '#00F0FF',
          marginBottom: '15px',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        {title}
      </div>
      <div
        style={{
          fontSize: '20px',
          color: '#ffffff',
          lineHeight: 1.6,
          opacity: 0.9,
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        {description}
      </div>
    </div>
  );
};

export const FeaturesScene: React.FC = () => {
  const frame = useCurrentFrame();

  // Title animation
  const titleOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const titleY = interpolate(frame, [0, 20], [-30, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)',
        padding: '80px',
      }}
    >
      {/* Background pattern */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: `radial-gradient(circle at 20% 50%, #0047FF10 0%, transparent 50%),
                            radial-gradient(circle at 80% 50%, #00F0FF10 0%, transparent 50%)`,
        }}
      />

      {/* Title */}
      <div
        style={{
          opacity: titleOpacity,
          transform: `translateY(${titleY}px)`,
          marginBottom: '60px',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            fontSize: '56px',
            fontWeight: 900,
            background: 'linear-gradient(135deg, #FAFF00 0%, #00F0FF 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            fontFamily: 'system-ui, -apple-system, sans-serif',
          }}
        >
          Why Choose o-mega?
        </div>
      </div>

      {/* Features grid */}
      <div
        style={{
          display: 'flex',
          gap: '40px',
          justifyContent: 'center',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          maxWidth: '1400px',
          margin: '0 auto',
        }}
      >
        <Feature
          title="AI Workers"
          description="Autonomous agents that handle tasks intelligently, 24/7"
          icon="🤖"
          delay={25}
        />
        <Feature
          title="Scalability"
          description="Scale your workforce instantly without hiring constraints"
          icon="📈"
          delay={40}
        />
        <Feature
          title="Automation"
          description="Automate complex workflows with intelligent AI agents"
          icon="⚡"
          delay={55}
        />
      </div>

      {/* Bottom accent */}
      <div
        style={{
          position: 'absolute',
          bottom: '50px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '80%',
          height: '2px',
          background: 'linear-gradient(90deg, transparent, #0047FF, #00F0FF, transparent)',
          opacity: interpolate(frame, [70, 90], [0, 0.5], {
            extrapolateLeft: 'clamp',
            extrapolateRight: 'clamp',
          }),
        }}
      />
    </AbsoluteFill>
  );
};
