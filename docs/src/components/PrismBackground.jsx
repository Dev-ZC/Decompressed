import React, { useEffect, useRef } from 'react';

export default function PrismBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let time = 0;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const drawPrism = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      // Create gradient background
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, 'rgba(139, 92, 246, 0.05)');
      gradient.addColorStop(0.5, 'rgba(59, 130, 246, 0.05)');
      gradient.addColorStop(1, 'rgba(236, 72, 153, 0.05)');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Draw animated prism lines
      const numLines = 50;
      for (let i = 0; i < numLines; i++) {
        const progress = (i / numLines + time * 0.1) % 1;
        const x1 = width * progress;
        const y1 = 0;
        const x2 = width * (1 - progress);
        const y2 = height;

        const hue = (progress * 360 + time * 50) % 360;
        const opacity = Math.sin(progress * Math.PI) * 0.3;

        ctx.strokeStyle = `hsla(${hue}, 70%, 60%, ${opacity})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      // Draw diagonal lines
      for (let i = 0; i < numLines; i++) {
        const progress = (i / numLines - time * 0.1) % 1;
        const x1 = 0;
        const y1 = height * progress;
        const x2 = width;
        const y2 = height * (1 - progress);

        const hue = (progress * 360 - time * 50) % 360;
        const opacity = Math.sin(progress * Math.PI) * 0.3;

        ctx.strokeStyle = `hsla(${hue}, 70%, 60%, ${opacity})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      time += 0.01;
      animationFrameId = requestAnimationFrame(drawPrism);
    };

    drawPrism();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        pointerEvents: 'none',
        maskImage: 'linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 10%, rgba(0,0,0,1) 70%, rgba(0,0,0,0) 100%), linear-gradient(to right, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 15%, rgba(0,0,0,1) 85%, rgba(0,0,0,0) 100%)',
        WebkitMaskImage: 'linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 10%, rgba(0,0,0,1) 70%, rgba(0,0,0,0) 100%), linear-gradient(to right, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 15%, rgba(0,0,0,1) 85%, rgba(0,0,0,0) 100%)',
        maskComposite: 'intersect',
        WebkitMaskComposite: 'source-in',
      }}
    />
  );
}
