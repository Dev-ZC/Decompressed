import React, { useEffect, useState } from 'react';

const CYCLES_PER_LETTER = 2;
const SHUFFLE_TIME = 50;
const CHARS = "!@#$%^&*():{};|,.<>/?";

export default function DecryptedText({ text, className = "" }) {
  const [displayText, setDisplayText] = useState(text);

  useEffect(() => {
    let intervalId = null;
    let currentIteration = 0;

    const scramble = () => {
      intervalId = setInterval(() => {
        setDisplayText((prev) =>
          text
            .split("")
            .map((char, index) => {
              if (index < currentIteration) {
                return text[index];
              }
              return CHARS[Math.floor(Math.random() * CHARS.length)];
            })
            .join("")
        );

        if (currentIteration >= text.length) {
          clearInterval(intervalId);
        }

        currentIteration += 1 / CYCLES_PER_LETTER;
      }, SHUFFLE_TIME);
    };

    scramble();

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [text]);

  return <span className={className}>{displayText}</span>;
}
