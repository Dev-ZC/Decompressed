import { useEffect, useRef } from 'react';
import './PrismaticBurst.css';

export default function PrismaticBurst({ onComplete, className = '' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const particles = [];
    const particleCount = 80;
    let animationId;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();

    class Particle {
      constructor() {
        this.reset();
      }

      reset() {
        this.x = canvas.width / 2;
        this.y = canvas.height / 2;
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 3 + 2;
        this.vx = Math.cos(angle) * speed;
        this.vy = Math.sin(angle) * speed;
        this.life = 1;
        this.decay = Math.random() * 0.015 + 0.01;
        this.size = Math.random() * 3 + 2;
        this.hue = Math.random() * 60 + 240; // Blue to purple range
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;
        this.life -= this.decay;
        this.size *= 0.98;
      }

      draw() {
        ctx.save();
        ctx.globalAlpha = this.life;
        const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.size);
        gradient.addColorStop(0, `hsla(${this.hue}, 80%, 70%, ${this.life})`);
        gradient.addColorStop(1, `hsla(${this.hue}, 80%, 50%, 0)`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    // Create particles
    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }

    let startTime = Date.now();
    const duration = 1500; // 1.5 seconds

    const animate = () => {
      const elapsed = Date.now() - startTime;
      
      if (elapsed > duration) {
        if (onComplete) onComplete();
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((particle) => {
        particle.update();
        particle.draw();
      });

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [onComplete]);

  return <canvas ref={canvasRef} className={`prismatic-burst ${className}`} />;
}
