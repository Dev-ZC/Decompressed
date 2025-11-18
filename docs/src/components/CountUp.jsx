import { useEffect, useRef } from 'react';
import { animate } from 'framer-motion';

export default function CountUp({ from = 0, to, duration = 2, delay = 0, className = '' }) {
  const nodeRef = useRef(null);

  useEffect(() => {
    const node = nodeRef.current;
    if (!node) return;

    const controls = animate(from, to, {
      duration,
      delay,
      onUpdate(value) {
        node.textContent = value.toFixed(1);
      }
    });

    return () => controls.stop();
  }, [from, to, duration, delay]);

  return <span ref={nodeRef} className={className} />;
}
