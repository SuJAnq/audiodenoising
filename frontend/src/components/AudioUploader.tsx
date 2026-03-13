import {
  useCallback,
  useRef,
  useState,
  type DragEvent,
  type ChangeEvent,
} from "react";

const ACCEPTED_TYPES = [
  "audio/wav",
  "audio/mpeg",
  "audio/mp3",
  "audio/flac",
  "audio/ogg",
  "audio/webm",
  "audio/x-wav",
];
const MAX_SIZE_MB = 50;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function AudioUploader({
  file,
  onFileSelect,
  onRemove,
  disabled,
}: {
  file: File | null;
  onFileSelect: (f: File) => void;
  onRemove: () => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validate = useCallback((f: File): string | null => {
    if (
      !ACCEPTED_TYPES.includes(f.type) &&
      !f.name.match(/\.(wav|mp3|flac|ogg|webm)$/i)
    ) {
      return "Unsupported format. Please upload WAV, MP3, FLAC, OGG, or WebM.";
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large (max ${MAX_SIZE_MB} MB).`;
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (f: File) => {
      const err = validate(f);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      onFileSelect(f);
    },
    [validate, onFileSelect],
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) handleFile(f);
      // Reset so the same file can be re-selected
      e.target.value = "";
    },
    [handleFile],
  );

  const openFilePicker = useCallback(() => {
    if (disabled) return;
    inputRef.current?.click();
  }, [disabled]);

  if (file) {
    return (
      <div className="upload-zone upload-zone--has-file">
        <div className="file-info__icon">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M9 18V5l12-2v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
        </div>
        <div className="file-info__details">
          <div className="file-info__name">{file.name}</div>
          <div className="file-info__meta">{formatSize(file.size)}</div>
        </div>
        {!disabled && (
          <button
            className="file-info__remove"
            onClick={onRemove}
            title="Remove file"
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}
      </div>
    );
  }

  return (
    <div>
      <div
        className={`upload-zone ${dragActive ? "upload-zone--active" : ""}`}
        onDragOver={(e) => {
          if (disabled) return;
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        onClick={(e) => {
          const target = e.target as HTMLElement;
          if (target.tagName.toLowerCase() === "input") return;
          openFilePicker();
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            openFilePicker();
          }
        }}
      >
        <div className="upload-zone__icon">
          {dragActive ? (
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
            </svg>
          ) : (
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" y1="19" x2="12" y2="23" />
              <line x1="8" y1="23" x2="16" y2="23" />
            </svg>
          )}
        </div>
        <div className="upload-zone__label">
          {dragActive ? "Drop your audio file here" : "Upload noisy audio"}
        </div>
        <div className="upload-zone__hint">
          Drag &amp; drop or click to browse — WAV, MP3, FLAC, OGG, WebM (max{" "}
          {MAX_SIZE_MB} MB)
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.webm,audio/*"
          onChange={onChange}
          onClick={(e) => e.stopPropagation()}
          tabIndex={-1}
          style={{ display: "none" }}
        />
      </div>
      {error && (
        <div className="status status--error" style={{ marginTop: "0.75rem" }}>
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}
