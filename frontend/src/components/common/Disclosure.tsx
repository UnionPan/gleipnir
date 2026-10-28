import React, { useState } from 'react';

/** A toggle that mounts its content only when open (charts measure their width on mount). */
const Disclosure: React.FC<{ title: string; hint?: string; children: React.ReactNode }> = ({ title, hint, children }) => {
  const [open, setOpen] = useState(false);
  return (
    <details className="lab-disclosure" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>{title}{hint && <span>{hint}</span>}</summary>
      {open && children}
    </details>
  );
};

export default Disclosure;
