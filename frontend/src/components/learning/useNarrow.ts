import { useEffect, useState } from 'react';

const QUERY = '(max-width: 600px)';

/** True on phone-width screens, where line charts trade end labels for a legend above. */
export function useNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => window.matchMedia(QUERY).matches);
  useEffect(() => {
    const media = window.matchMedia(QUERY);
    const update = () => setNarrow(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  return narrow;
}
