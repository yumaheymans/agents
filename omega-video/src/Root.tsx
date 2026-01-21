import { Composition, registerRoot } from 'remotion';
import { OmegaVideo } from './OmegaVideo';

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="OmegaVideo"
        component={OmegaVideo}
        durationInFrames={450}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};

registerRoot(Root);
