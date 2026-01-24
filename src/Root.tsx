import { Composition } from 'remotion';
import { FlashyTextComposition } from './Composition';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="FlashyText"
        component={FlashyTextComposition}
        durationInFrames={720} // 24 seconds at 30fps (8 animations × 3 seconds each)
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
