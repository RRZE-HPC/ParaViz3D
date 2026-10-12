let mediaRecorder = null;
let recordedChunks = [];
let recording = false;
let onStopCallback = null;

const CANDIDATE_MIME_TYPES = [
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];

function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return null;
  for (const type of CANDIDATE_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}

export function isRecordingSupported(canvas) {
  return typeof MediaRecorder !== 'undefined' &&
         typeof canvas?.captureStream === 'function';
}

export function isRecording() {
  return recording;
}

export function startRecording(canvas, { fps = 0, onStop } = {}) {
  if (!isRecordingSupported(canvas) || recording) return false;

  const stream = fps > 0 ? canvas.captureStream(fps) : canvas.captureStream();
  const mimeType = pickMimeType();
  const options = { videoBitsPerSecond: 8_000_000 };
  if (mimeType) options.mimeType = mimeType;

  try {
    mediaRecorder = new MediaRecorder(stream, options);
  } catch (err) {
    alert(`Recording is not supported in this browser: ${err.message}`);
    return false;
  }

  recordedChunks = [];
  onStopCallback = onStop || null;

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    recording = false;
    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'video/webm' });
    recordedChunks = [];
    onStopCallback?.(blob);
  };
  mediaRecorder.onerror = (e) => console.error('MediaRecorder error:', e.error);

  mediaRecorder.start(1000);
  recording = true;
  return true;
}

export function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
}

export function downloadBlob(blob, filenamePrefix = 'paraviz3d-recording') {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  a.href = url;
  a.download = `${filenamePrefix}-${ts}.webm`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
