'use client';

import type { ChangeEvent, CSSProperties, DragEvent } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Activity,
  BarChart3,
  Brain,
  Check,
  CircleCheck,
  CircleX,
  Cpu,
  Database,
  FileText,
  History,
  LayoutDashboard,
  LoaderCircle,
  Pencil,
  Play,
  RotateCcw,
  Search,
  Server,
  Settings,
  Shield,
  Square,
  Trash2,
  TrendingDown,
  TrendingUp,
  Upload,
  X,
} from 'lucide-react';

type Page = 'dashboard' | 'upload' | 'review' | 'history' | 'settings' | 'local';
type UploadStatus = 'queued' | 'uploading' | 'done' | 'error';
type HistoryFilter = 'all' | 'extracted' | 'unresolved' | 'corrected';

interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface FieldResult {
  field_name?: string;
  value: unknown;
  status: string;
  confidence?: number;
  source_bbox?: BoundingBox | null;
  raw_text?: string | null;
  validation_message?: string | null;
}

interface ExtractionMetadata {
  extraction_id: string;
  timestamp?: string;
  vlm_model?: string;
  llm_model?: string;
  processing_time_ms?: number;
}

interface ExtractionResult {
  metadata: ExtractionMetadata;
  document_type?: string;
  fields?: Record<string, FieldResult>;
  unresolved_count?: number;
  validation_failure_count?: number;
  corrected_count?: number;
  overall_confidence?: number;
  __previewUrl?: string;
  __fileName?: string;
  __fileType?: string;
}

interface UploadJob {
  id: number;
  file: File;
  status: UploadStatus;
  uploadProgress: number;
  result: ExtractionResult | null;
  errorMsg: string;
  xhr: XMLHttpRequest | null;
  startTime: number | null;
  endTime: number | null;
}

interface LocalTrainingStatus {
  state: string;
  running: boolean;
  pid?: number | null;
  started_at?: number | null;
  processed: number;
  run_processed: number;
  stored: number;
  skipped: number;
  pending_batch: number;
  total_images: number;
  progress_pct: number;
  processed_progress_pct: number;
  indexed_progress_pct: number;
  images_per_second: number;
  updated_at?: string | null;
  updated_at_epoch?: number | null;
  heartbeat_age_seconds?: number | null;
  current_file?: string | null;
  current_index?: number | null;
  last_event?: string | null;
  learning_score: number;
  avg_quality_score: number;
  avg_ocr_chars: number;
  ocr_coverage_rate: number;
  visual_hash_coverage_rate: number;
  vector_index_enabled: boolean;
  ocr_engine?: string;
  ocr_batch_size?: number;
  surya_device?: string;
  surya_recognition_batch_size?: number;
  surya_detector_batch_size?: number;
  surya_task_name?: string;
  top_labels: Record<string, number>;
  recent_log: string[];
}

interface TrainingPoint {
  time: number;
  score: number;
  processed: number;
  stored: number;
}

const allowedTypes = new Set([
  'image/png',
  'image/jpeg',
  'image/jpg',
  'image/tiff',
  'image/bmp',
  'application/pdf',
  'text/plain',
  'text/csv',
  'text/tab-separated-values',
  'text/markdown',
  'text/html',
  'application/json',
  'application/xml',
  'text/xml',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
]);
const allowedExtensions = /\.(png|jpe?g|tiff?|bmp|pdf|txt|csv|tsv|md|json|xml|html?|docx)$/i;

const navItems: Array<{ id: Page; label: string; Icon: typeof LayoutDashboard }> = [
  { id: 'dashboard', label: 'Dashboard', Icon: LayoutDashboard },
  { id: 'upload', label: 'Upload', Icon: Upload },
  { id: 'review', label: 'Review', Icon: Search },
  { id: 'history', label: 'History', Icon: History },
  { id: 'local', label: 'Local', Icon: Brain },
  { id: 'settings', label: 'Settings', Icon: Settings },
];

const defaultApiUrl = () => {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8000';
  return window.location.port === '3000' ? 'http://127.0.0.1:8000' : window.location.origin;
};

const normalizeApiUrl = (value: string) => value.replace('http://localhost:8000', 'http://127.0.0.1:8000');

function resultFields(result: ExtractionResult | null): [string, FieldResult][] {
  if (!result?.fields || typeof result.fields !== 'object') return [];
  return Object.entries(result.fields);
}

function formatSize(bytes: number): string {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(Math.round(value || 0));
}

function formatPercent(value: number, digits = 1): string {
  return `${(clamp(value, 0, 1) * 100).toFixed(digits)}%`;
}

function formatDate(value?: string): string {
  if (!value) return '--';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--';
  return date.toLocaleString('de-DE', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

function statusBadge(status: string, confidence?: number) {
  const label =
    status === 'EXTRACTED'
      ? 'Extracted'
      : status === 'LOW_CONFIDENCE'
        ? 'Low Confidence'
        : status === 'UNRESOLVED'
          ? 'Unresolved'
          : status === 'VALIDATION_FAILURE'
            ? 'Validation Fail'
            : status === 'CORRECTED'
              ? 'Corrected'
              : status || 'Unknown';
  const tone =
    status === 'EXTRACTED'
      ? 'green'
      : status === 'CORRECTED'
        ? 'blue'
        : status === 'UNRESOLVED' || status === 'VALIDATION_FAILURE'
          ? 'red'
          : 'amber';
  const suffix = typeof confidence === 'number' ? ` ${Math.round(clamp(confidence, 0, 1) * 100)}%` : '';
  return <span className={`badge ${tone}`}>{label}{suffix}</span>;
}

function docType(result: ExtractionResult): string {
  return result.document_type || 'Unknown document';
}

function parseServerError(xhr: XMLHttpRequest): string {
  try {
    const body = JSON.parse(xhr.responseText) as { message?: string; detail?: string; error_code?: string };
    return body.message || body.detail || body.error_code || `HTTP ${xhr.status}`;
  } catch {
    return xhr.statusText || `HTTP ${xhr.status}`;
  }
}

function bboxStyle(bbox: BoundingBox): CSSProperties {
  const left = clamp(bbox.x1 * 100, 0, 100);
  const top = clamp(bbox.y1 * 100, 0, 100);
  const width = clamp((bbox.x2 - bbox.x1) * 100, 0, 100 - left);
  const height = clamp((bbox.y2 - bbox.y1) * 100, 0, 100 - top);
  return {
    '--bbox-left': `${left}%`,
    '--bbox-top': `${top}%`,
    '--bbox-width': `${width}%`,
    '--bbox-height': `${height}%`,
  } as CSSProperties;
}

export function DashboardApp({ initialPage = 'upload' }: { initialPage?: Page }) {
  const router = useRouter();
  const [page, setPage] = useState<Page>(initialPage);
  const [apiUrl, setApiUrl] = useState(defaultApiUrl);
  const [settingsValue, setSettingsValue] = useState(defaultApiUrl);
  const [settingsSaved, setSettingsSaved] = useState(false);
  const [uploadNotice, setUploadNotice] = useState('');
  const [uploadJobs, setUploadJobs] = useState<UploadJob[]>([]);
  const [extractionHistory, setExtractionHistory] = useState<ExtractionResult[]>([]);
  const [currentResult, setCurrentResult] = useState<ExtractionResult | null>(null);
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [highlightedField, setHighlightedField] = useState<string | null>(null);
  const [correctingField, setCorrectingField] = useState<string | null>(null);
  const [historySearch, setHistorySearch] = useState('');
  const [historyFilter, setHistoryFilter] = useState<HistoryFilter>('all');
  const [dragActive, setDragActive] = useState(false);
  const [nextJobId, setNextJobId] = useState(1);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const jobsRef = useRef<UploadJob[]>([]);
  const processingRef = useRef(false);
  const processNextRef = useRef<() => void>(() => undefined);

  const setJobsSafely = useCallback((updater: (jobs: UploadJob[]) => UploadJob[]) => {
    setUploadJobs((previous) => {
      const next = updater(previous);
      jobsRef.current = next;
      return next;
    });
  }, []);

  useEffect(() => {
    const stored = window.localStorage.getItem('abe_api');
    const resolved = normalizeApiUrl(stored || defaultApiUrl());
    if (stored && stored !== resolved) {
      window.localStorage.setItem('abe_api', resolved);
    }
    setApiUrl(resolved);
    setSettingsValue(resolved);
  }, []);

  useEffect(() => {
    setPage(initialPage);
  }, [initialPage]);

  useEffect(() => {
    jobsRef.current = uploadJobs;
  }, [uploadJobs]);

  useEffect(() => {
    return () => {
      const urls = new Set<string>();
      jobsRef.current.forEach((job) => {
        if (job.xhr && job.status === 'uploading') job.xhr.abort();
        if (job.result?.__previewUrl) urls.add(job.result.__previewUrl);
      });
      extractionHistory.forEach((result) => {
        if (result.__previewUrl) urls.add(result.__previewUrl);
      });
      urls.forEach((url) => URL.revokeObjectURL(url));
    };
  }, [extractionHistory]);

  const updateJob = useCallback(
    (jobId: number, patch: Partial<UploadJob>) => {
      setJobsSafely((jobs) => jobs.map((job) => (job.id === jobId ? { ...job, ...patch } : job)));
    },
    [setJobsSafely],
  );

  const executeUpload = useCallback(
    (job: UploadJob) => {
      updateJob(job.id, {
        status: 'uploading',
        uploadProgress: 0,
        startTime: Date.now(),
        endTime: null,
        errorMsg: '',
      });

      const form = new FormData();
      form.append('file', job.file, job.file.name);

      const xhr = new XMLHttpRequest();
      updateJob(job.id, { xhr });

      xhr.upload.addEventListener('progress', (event) => {
        if (!event.lengthComputable) return;
        updateJob(job.id, { uploadProgress: Math.round((event.loaded / event.total) * 100) });
      });

      xhr.addEventListener('load', () => {
        const endTime = Date.now();
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const body = JSON.parse(xhr.responseText) as unknown;
            const result = (
              typeof body === 'object' && body !== null && 'result' in body
                ? (body as { result?: unknown }).result
                : body
            ) as ExtractionResult;
            if (!result.metadata?.extraction_id || !result.fields || typeof result.fields !== 'object') {
              throw new Error('Server returned an incomplete extraction result.');
            }

            const enriched: ExtractionResult = {
              ...result,
              __previewUrl: URL.createObjectURL(job.file),
              __fileName: job.file.name,
              __fileType: job.file.type,
            };

            setExtractionHistory((history) => [enriched, ...history]);
            setCurrentResult(enriched);
            updateJob(job.id, { status: 'done', uploadProgress: 100, result: enriched, endTime, xhr: null });
          } catch (error) {
            updateJob(job.id, {
              status: 'error',
              endTime,
              xhr: null,
              errorMsg: error instanceof Error ? error.message : 'Could not read the server response.',
            });
          }
        } else {
          updateJob(job.id, { status: 'error', endTime, xhr: null, errorMsg: parseServerError(xhr) });
        }
        processingRef.current = false;
        window.setTimeout(() => processNextRef.current(), 250);
      });

      xhr.addEventListener('error', () => {
        updateJob(job.id, {
          status: 'error',
          endTime: Date.now(),
          xhr: null,
          errorMsg: 'Network error. Check that the FastAPI server is running.',
        });
        processingRef.current = false;
        window.setTimeout(() => processNextRef.current(), 350);
      });

      xhr.addEventListener('abort', () => {
        updateJob(job.id, { status: 'error', endTime: Date.now(), xhr: null, errorMsg: 'Upload cancelled.' });
        processingRef.current = false;
        window.setTimeout(() => processNextRef.current(), 350);
      });

      xhr.open('POST', `${apiUrl.replace(/\/$/, '')}/api/v1/extract`);
      xhr.send(form);
    },
    [apiUrl, updateJob],
  );

  const processNext = useCallback(() => {
    if (processingRef.current) return;
    const next = jobsRef.current.find((job) => job.status === 'queued');
    if (!next) return;
    processingRef.current = true;
    executeUpload(next);
  }, [executeUpload]);

  useEffect(() => {
    processNextRef.current = processNext;
  }, [processNext]);

  const addFilesToQueue = useCallback(
    (incoming: File[]) => {
      const files = [...incoming];
      if (!files.length) return;

      let rejected = 0;
      let duplicate = 0;
      const accepted: UploadJob[] = [];
      let idCursor = nextJobId;

      for (const file of files) {
        const validType = allowedTypes.has(file.type) || allowedExtensions.test(file.name);
        if (!validType) {
          rejected += 1;
          continue;
        }

        const duplicateFile = jobsRef.current.some(
          (job) => job.file.name === file.name && job.file.size === file.size && job.file.lastModified === file.lastModified,
        );
        if (duplicateFile) {
          duplicate += 1;
          continue;
        }

        accepted.push({
          id: idCursor,
          file,
          status: 'queued',
          uploadProgress: 0,
          result: null,
          errorMsg: '',
          xhr: null,
          startTime: null,
          endTime: null,
        });
        idCursor += 1;
      }

      setNextJobId(idCursor);
      if (accepted.length) {
        setJobsSafely((jobs) => [...jobs, ...accepted]);
      }

      const notes: string[] = [];
      if (rejected) notes.push(`${rejected} unsupported file${rejected === 1 ? '' : 's'} skipped`);
      if (duplicate) notes.push(`${duplicate} duplicate file${duplicate === 1 ? '' : 's'} skipped`);
      setUploadNotice(notes.join('. '));

      window.setTimeout(() => processNextRef.current(), 0);
    },
    [nextJobId, setJobsSafely],
  );

  const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.currentTarget.files ?? []);
    event.currentTarget.value = '';
    addFilesToQueue(files);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    addFilesToQueue(Array.from(event.dataTransfer.files ?? []));
  };

  const removeJob = (jobId: number) => {
    const job = jobsRef.current.find((item) => item.id === jobId);
    if (job?.xhr && job.status === 'uploading') job.xhr.abort();
    setJobsSafely((jobs) => jobs.filter((item) => item.id !== jobId));
  };

  const retryJob = (jobId: number) => {
    setJobsSafely((jobs) =>
      jobs.map((job) =>
        job.id === jobId
          ? { ...job, status: 'queued', uploadProgress: 0, result: null, errorMsg: '', xhr: null, startTime: null, endTime: null }
          : job,
      ),
    );
    window.setTimeout(() => processNextRef.current(), 0);
  };

  const clearJobs = (status: UploadStatus) => {
    setJobsSafely((jobs) => jobs.filter((job) => job.status !== status));
  };

  const openReview = (id: string) => {
    const result = extractionHistory.find((item) => item.metadata.extraction_id === id);
    if (result) setCurrentResult(result);
    setPage('review');
  };

  const saveCorrection = (fieldName: string) => {
    if (!currentResult?.fields?.[fieldName]) return;
    const input = document.getElementById('corrValue') as HTMLInputElement | null;
    const correctedValue = input?.value ?? '';
    const originalValue = currentResult.fields[fieldName].value;

    void fetch(`${apiUrl.replace(/\/$/, '')}/api/v1/corrections`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        extraction_id: currentResult.metadata.extraction_id,
        field_name: fieldName,
        corrected_value: correctedValue,
        original_value: originalValue,
      }),
    }).catch(() => undefined);

    const updateResult = (result: ExtractionResult): ExtractionResult => ({
      ...result,
      corrected_count: (result.corrected_count || 0) + 1,
      fields: {
        ...(result.fields || {}),
        [fieldName]: {
          ...result.fields?.[fieldName],
          value: correctedValue,
          status: 'CORRECTED',
          confidence: 1,
        } as FieldResult,
      },
    });

    const updated = updateResult(currentResult);
    setCurrentResult(updated);
    setExtractionHistory((historyRows) =>
      historyRows.map((row) => (row.metadata.extraction_id === currentResult.metadata.extraction_id ? updateResult(row) : row)),
    );
    setCorrectingField(null);
    setSelectedField(fieldName);
  };

  const filteredHistory = useMemo(() => {
    let rows = [...extractionHistory];
    if (historyFilter === 'unresolved') rows = rows.filter((item) => (item.unresolved_count || 0) > 0);
    if (historyFilter === 'corrected') rows = rows.filter((item) => (item.corrected_count || 0) > 0);
    if (historyFilter === 'extracted') rows = rows.filter((item) => (item.unresolved_count || 0) === 0);

    const query = historySearch.trim().toLowerCase();
    if (query) {
      rows = rows.filter((item) => docType(item).toLowerCase().includes(query) || item.metadata.extraction_id.toLowerCase().includes(query));
    }
    return rows;
  }, [extractionHistory, historyFilter, historySearch]);

  const navigate = (nextPage: Page) => {
    setSettingsSaved(false);
    setPage(nextPage);
    if (nextPage === 'local') {
      router.push('/dashboard/local');
    } else if (page === 'local') {
      router.push('/dashboard');
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><FileText size={18} /></div>
          <div>
            <h1>Digitalisierung</h1>
            <p>ABE System</p>
          </div>
        </div>
        <nav className="nav" aria-label="Main">
          {navItems.map(({ id, label, Icon }) => (
            <button key={id} className={`nav-button ${page === id ? 'active' : ''}`} onClick={() => navigate(id)} type="button">
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-status">
          <div className="status-row"><span className="status-dot" /><span>System Online</span></div>
          <p className="fine">Next.js App Router</p>
        </div>
      </aside>

      <main className="main">
        {page === 'dashboard' && (
          <DashboardPage extractionHistory={extractionHistory} onOpenReview={openReview} onNavigate={navigate} />
        )}

        {page === 'local' && <LocalTrainingPage apiUrl={apiUrl} />}

        {page === 'upload' && (
          <section className="page narrow">
            <div className="page-header">
              <div>
                <h2 className="page-title">Upload Documents</h2>
                <p className="page-copy">Drop files here or use the file picker. Images and PDFs are validated before upload.</p>
              </div>
            </div>
            <div
              className={`dropzone ${dragActive ? 'drag-active' : ''}`}
              onClick={(event) => {
                if ((event.target as HTMLElement).closest('button,input')) return;
                fileInputRef.current?.click();
              }}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={(event) => {
                event.preventDefault();
                if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragActive(false);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                className="sr-only"
                type="file"
                accept="image/png,image/jpeg,image/jpg,image/tiff,image/bmp,application/pdf,text/plain,text/csv,text/tab-separated-values,text/markdown,text/html,application/json,application/xml,text/xml,application/vnd.openxmlformats-officedocument.wordprocessingml.document,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.pdf,.txt,.csv,.tsv,.md,.json,.xml,.html,.htm,.docx"
                multiple
                onChange={handleFileInputChange}
              />
              <div className="dropzone-content">
                <div className="icon-box large-icon"><Upload size={24} /></div>
                <div>
                  <h3>Upload administrative forms</h3>
                  <p>PNG, JPG, TIFF, BMP, or PDF. Backend limit: 50 MB per file.</p>
                </div>
                <button className="primary-button" onClick={() => fileInputRef.current?.click()} type="button">
                  <Upload size={17} /> Choose files
                </button>
              </div>
            </div>
            {uploadNotice && <p className="upload-note">{uploadNotice}</p>}
            <UploadJobs jobs={uploadJobs} onClear={clearJobs} onOpenReview={openReview} onRemove={removeJob} onRetry={retryJob} />
          </section>
        )}

        {page === 'review' && (
          <ReviewPage
            result={currentResult}
            selectedField={selectedField}
            highlightedField={highlightedField}
            correctingField={correctingField}
            onCloseCorrection={() => setCorrectingField(null)}
            onCorrectField={(field) => {
              setSelectedField(field);
              setCorrectingField(field);
            }}
            onHighlightField={setHighlightedField}
            onNavigate={navigate}
            onSaveCorrection={saveCorrection}
            onSelectField={setSelectedField}
          />
        )}

        {page === 'history' && (
          <HistoryPage
            allRows={extractionHistory}
            filteredRows={filteredHistory}
            filter={historyFilter}
            search={historySearch}
            onFilter={setHistoryFilter}
            onOpenReview={openReview}
            onSearch={setHistorySearch}
          />
        )}

        {page === 'settings' && (
          <SettingsPage
            apiUrl={settingsValue}
            saved={settingsSaved}
            onApiUrlChange={setSettingsValue}
            onReset={() => {
              const resolved = defaultApiUrl();
              setApiUrl(resolved);
              setSettingsValue(resolved);
              window.localStorage.removeItem('abe_api');
              setSettingsSaved(true);
            }}
            onSave={() => {
              const normalized = normalizeApiUrl(settingsValue.replace(/\/$/, ''));
              setApiUrl(normalized);
              setSettingsValue(normalized);
              window.localStorage.setItem('abe_api', normalized);
              setSettingsSaved(true);
            }}
          />
        )}
      </main>
    </div>
  );
}

function DashboardPage({
  extractionHistory,
  onNavigate,
  onOpenReview,
}: {
  extractionHistory: ExtractionResult[];
  onNavigate: (page: Page) => void;
  onOpenReview: (id: string) => void;
}) {
  const total = extractionHistory.length;
  const avgConfidence = total
    ? Math.round((extractionHistory.reduce((sum, item) => sum + (item.overall_confidence || 0), 0) / total) * 100)
    : 0;
  const unresolved = extractionHistory.reduce((sum, item) => sum + (item.unresolved_count || 0), 0);
  const fieldTotal = extractionHistory.reduce((sum, item) => sum + resultFields(item).length, 0);
  const cards = [
    { label: 'Documents', value: total, Icon: FileText },
    { label: 'Avg Confidence', value: `${avgConfidence}%`, Icon: Activity },
    { label: 'Unresolved', value: unresolved, Icon: Search },
    { label: 'Fields Read', value: fieldTotal, Icon: Database },
  ];

  return (
    <section className="page">
      <div className="page-header">
        <div>
          <h2 className="page-title">Dashboard</h2>
          <p className="page-copy">Operational overview for local document extraction.</p>
        </div>
        <div className="header-actions">
          <button className="ghost-button" onClick={() => onNavigate('local')} type="button"><Brain size={17} /> Switch to local</button>
          <button className="primary-button" onClick={() => onNavigate('upload')} type="button"><Upload size={17} /> Upload</button>
        </div>
      </div>
      <div className="grid stats">
        {cards.map(({ label, value, Icon }) => (
          <div className="stat-card" key={label}>
            <div className="stat-top"><div className="icon-box"><Icon size={18} /></div><p className="stat-value">{value}</p></div>
            <p className="stat-label">{label}</p>
          </div>
        ))}
      </div>
      <div className="panel">
        <div className="panel-header"><h3 className="panel-title">Recent Extractions</h3><span className="fine">{total} total</span></div>
        {total === 0 ? (
          <div className="empty-state">
            <div className="icon-box large-icon"><Upload size={24} /></div>
            <p>No documents processed yet.</p>
            <button className="ghost-button" onClick={() => onNavigate('upload')} type="button">Choose files</button>
          </div>
        ) : (
          <div className="rows">{extractionHistory.slice(0, 8).map((item) => <RecentRow key={item.metadata.extraction_id} item={item} onOpenReview={onOpenReview} />)}</div>
        )}
      </div>
    </section>
  );
}

function LocalTrainingPage({ apiUrl }: { apiUrl: string }) {
  const [status, setStatus] = useState<LocalTrainingStatus | null>(null);
  const [points, setPoints] = useState<TrainingPoint[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);

  const endpoint = apiUrl.replace(/\/$/, '');

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${endpoint}/api/v1/local-training/status`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const next = (await response.json()) as LocalTrainingStatus;
      setStatus(next);
      setError('');
      setPoints((history) => {
        const point = {
          time: Date.now(),
          score: next.learning_score || 0,
          processed: next.processed || 0,
          stored: next.stored || 0,
        };
        const last = history[history.length - 1];
        if (last && last.processed === point.processed && last.score === point.score) return history;
        return [...history.slice(-47), point];
      });
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Status request failed');
    }
  }, [endpoint]);

  useEffect(() => {
    void fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const timer = window.setInterval(() => {
      void fetchStatus();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, fetchStatus]);

  const mutateTraining = async (action: 'start' | 'stop') => {
    setBusy(true);
    setError('');
    try {
      const response = await fetch(`${endpoint}/api/v1/local-training/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: action === 'start' ? JSON.stringify({ batch_size: 32, max_seconds: 3300, plateau_window: 50000, heartbeat_seconds: 1 }) : undefined,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const next = (await response.json()) as LocalTrainingStatus;
      setStatus(next);
      await fetchStatus();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : `${action} failed`);
    } finally {
      setBusy(false);
    }
  };

  const current = status;
  const previousPoint = points.length > 1 ? points[points.length - 2] : null;
  const latestPoint = points[points.length - 1] || null;
  const scoreDelta = latestPoint && previousPoint ? latestPoint.score - previousPoint.score : 0;
  const processedCoverage = current?.total_images ? current.processed / current.total_images : 0;
  const indexedCoverage = current?.total_images ? current.stored / current.total_images : 0;
  const remaining = current ? Math.max(current.total_images - current.processed, 0) : 0;
  const etaSeconds = current?.images_per_second ? remaining / current.images_per_second : 0;
  const statusTone = current?.running ? 'green' : current?.state === 'stopped' ? 'amber' : 'blue';
  const topLabels = Object.entries(current?.top_labels || {}).slice(0, 6);
  const heartbeatAge = current?.heartbeat_age_seconds;
  const heartbeatLabel = heartbeatAge === null || heartbeatAge === undefined ? '--' : `${heartbeatAge.toFixed(1)}s ago`;

  const scoreCards = [
    { label: 'Learning Score', value: current ? current.learning_score.toFixed(1) : '--', Icon: BarChart3 },
    { label: 'Processed Coverage', value: current ? `${(current.processed_progress_pct ?? processedCoverage * 100).toFixed(2)}%` : '--', Icon: Database },
    { label: 'Learned Images', value: current ? formatNumber(current.stored) : '--', Icon: FileText },
    { label: 'Throughput', value: current ? `${current.images_per_second.toFixed(2)}/s` : '--', Icon: Activity },
  ];

  const metricRows = current
    ? [
        ['Total images', formatNumber(current.total_images)],
        ['Processed', formatNumber(current.processed)],
        ['This run', formatNumber(current.run_processed || 0)],
        ['Stored', formatNumber(current.stored)],
        ['Pending batch', formatNumber(current.pending_batch || 0)],
        ['Skipped', formatNumber(current.skipped)],
        ['Current file', current.current_file || '--'],
        ['Current index', current.current_index ? formatNumber(current.current_index) : '--'],
        ['Last event', current.last_event || '--'],
        ['Heartbeat', heartbeatLabel],
        ['Indexed coverage', `${(current.indexed_progress_pct ?? indexedCoverage * 100).toFixed(2)}%`],
        ['Average quality', formatPercent(current.avg_quality_score)],
        ['OCR coverage', formatPercent(current.ocr_coverage_rate)],
        ['Average OCR characters', current.avg_ocr_chars.toFixed(1)],
        ['OCR engine', current.ocr_engine || 'surya-ocr'],
        ['Surya device', current.surya_device || 'auto'],
        ['Surya task', current.surya_task_name || 'ocr_without_boxes'],
        ['OCR batch', `${current.ocr_batch_size || 32}`],
        ['Surya batch', `${current.surya_recognition_batch_size || 32}/${current.surya_detector_batch_size || 4}`],
        ['Visual hash coverage', formatPercent(current.visual_hash_coverage_rate)],
        ['Vector index', current.vector_index_enabled ? 'Enabled' : 'Fallback'],
        ['ETA at current rate', etaSeconds ? formatDuration(etaSeconds * 1000) : '--'],
      ]
    : [];

  return (
    <section className="page local-page">
      <div className="page-header">
        <div>
          <h2 className="page-title">Local Image Learning</h2>
          <p className="page-copy">RVL-CDIP image learning workspace.</p>
        </div>
        <div className="header-actions">
          <button className="ghost-button" onClick={() => setAutoRefresh((value) => !value)} type="button">
            <Activity size={17} /> {autoRefresh ? 'Live' : 'Paused'}
          </button>
          <button className="ghost-button" disabled={busy || !current?.running} onClick={() => void mutateTraining('stop')} type="button">
            <Square size={15} /> Stop
          </button>
          <button className="primary-button" disabled={busy || current?.running} onClick={() => void mutateTraining('start')} type="button">
            <Play size={17} /> Start local training
          </button>
        </div>
      </div>

      <div className="training-hero">
        <div>
          <div className="training-status-line">
            <span className={`badge ${statusTone}`}>{current?.state || 'loading'}</span>
            <span className="job-meta">{current?.updated_at || 'No progress file yet'}</span>
            <span className="job-meta">{current?.last_event || 'waiting'}</span>
          </div>
          <div className="training-score-row">
            <span className="training-score">{current ? current.learning_score.toFixed(1) : '--'}</span>
            <span className={`score-delta ${scoreDelta >= 0 ? 'up' : 'down'}`}>
              {scoreDelta >= 0 ? <TrendingUp size={17} /> : <TrendingDown size={17} />}
              {scoreDelta === 0 ? '0.0' : `${scoreDelta > 0 ? '+' : ''}${scoreDelta.toFixed(2)}`}
            </span>
          </div>
          <div className="wide-progress">
            <span style={{ '--progress': `${clamp(processedCoverage * 100, 0, 100)}%` } as CSSProperties} />
          </div>
          <div className="training-live-row">
            <span>Processing</span>
            <strong className="mono">{current?.current_file || '--'}</strong>
          </div>
        </div>
        <div className="training-summary">
          <span className="label">Managed PID</span>
          <strong className="mono">{current?.pid || '--'}</strong>
          <span className="label">Remaining to scan</span>
          <strong className="mono">{formatNumber(remaining)}</strong>
          <span className="label">Indexed</span>
          <strong className="mono">{current ? `${(current.indexed_progress_pct ?? indexedCoverage * 100).toFixed(2)}%` : '--'}</strong>
        </div>
      </div>

      {error && <div className="job-error">{error}</div>}

      <div className="grid stats">
        {scoreCards.map(({ label, value, Icon }) => (
          <div className="stat-card" key={label}>
            <div className="stat-top"><div className="icon-box"><Icon size={18} /></div><p className="stat-value">{value}</p></div>
            <p className="stat-label">{label}</p>
          </div>
        ))}
      </div>

      <div className="local-grid">
        <div className="panel">
          <div className="panel-header"><h3 className="panel-title">Learning Trend</h3><span className="fine">{points.length} samples</span></div>
          <div className="trend-chart">
            {points.length === 0 ? (
              <div className="empty-state"><p>No live samples yet.</p></div>
            ) : (
              points.map((point) => (
                <span
                  key={`${point.time}-${point.processed}`}
                  title={`${point.score.toFixed(1)} at ${formatNumber(point.stored)} images`}
                  style={{ '--bar': `${clamp(point.score, 4, 100)}%` } as CSSProperties}
                />
              ))
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header"><h3 className="panel-title">Stats Sheet</h3><span className="fine">live</span></div>
          <div className="metrics-table">
            {metricRows.map(([name, value]) => (
              <div className="metric-row" key={name}>
                <span>{name}</span>
                <strong className="mono">{value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header"><h3 className="panel-title">Corpus Mix</h3><span className="fine">{topLabels.length} labels</span></div>
          <div className="label-stack">
            {topLabels.length === 0 ? (
              <div className="empty-state compact"><p>No labels indexed.</p></div>
            ) : (
              topLabels.map(([label, count]) => (
                <div className="label-row" key={label}>
                  <span>{label}</span>
                  <span className="mono">{formatNumber(count)}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="panel log-panel">
          <div className="panel-header"><h3 className="panel-title">Training Log</h3><button className="link-button" onClick={() => void fetchStatus()} type="button">Refresh</button></div>
          <pre className="log-stream">{(current?.recent_log || []).slice(-22).join('\n') || 'Waiting for local training output.'}</pre>
        </div>
      </div>
    </section>
  );
}

function RecentRow({ item, onOpenReview }: { item: ExtractionResult; onOpenReview: (id: string) => void }) {
  return (
    <button className="recent-row" onClick={() => onOpenReview(item.metadata.extraction_id)} type="button">
      <span className="doc-cell">
        <span className="icon-box"><FileText size={17} /></span>
        <span><span className="doc-title">{docType(item)}</span><span className="job-meta">{item.metadata.extraction_id.slice(0, 8)}</span></span>
      </span>
      <span>{statusBadge((item.overall_confidence || 0) >= 0.7 ? 'EXTRACTED' : 'LOW_CONFIDENCE', item.overall_confidence)}</span>
      <span className="job-meta">{resultFields(item).length} fields</span>
    </button>
  );
}

function UploadJobs({
  jobs,
  onClear,
  onOpenReview,
  onRemove,
  onRetry,
}: {
  jobs: UploadJob[];
  onClear: (status: UploadStatus) => void;
  onOpenReview: (id: string) => void;
  onRemove: (id: number) => void;
  onRetry: (id: number) => void;
}) {
  if (!jobs.length) return null;
  const counts = {
    queued: jobs.filter((job) => job.status === 'queued').length,
    uploading: jobs.filter((job) => job.status === 'uploading').length,
    done: jobs.filter((job) => job.status === 'done').length,
    error: jobs.filter((job) => job.status === 'error').length,
  };

  return (
    <div className="job-list">
      <div className="job-summary">
        <p>{jobs.length} file{jobs.length === 1 ? '' : 's'} - {counts.queued} queued, {counts.uploading} active, {counts.done} done, {counts.error} failed</p>
        <div>
          {counts.done > 0 && <button className="link-button" onClick={() => onClear('done')} type="button">Clear done</button>}
          {counts.error > 0 && <button className="link-button" onClick={() => onClear('error')} type="button">Clear failed</button>}
        </div>
      </div>
      <div className="job-stack">
        {jobs.map((job) => (
          <UploadJobRow key={job.id} job={job} onOpenReview={onOpenReview} onRemove={onRemove} onRetry={onRetry} />
        ))}
      </div>
    </div>
  );
}

function UploadJobRow({
  job,
  onOpenReview,
  onRemove,
  onRetry,
}: {
  job: UploadJob;
  onOpenReview: (id: string) => void;
  onRemove: (id: number) => void;
  onRetry: (id: number) => void;
}) {
  const duration = job.endTime && job.startTime ? formatDuration(job.endTime - job.startTime) : '';
  const progress = clamp(job.uploadProgress, 0, 100);
  const StatusIcon = job.status === 'uploading' ? LoaderCircle : job.status === 'done' ? CircleCheck : job.status === 'error' ? CircleX : FileText;

  return (
    <article className={`job-card ${job.status}`}>
      <div className="job-main">
        <span className="icon-box"><StatusIcon className={job.status === 'uploading' ? 'spin' : ''} size={17} /></span>
        <div className="job-text">
          <p className="job-name">{job.file.name}</p>
          <p className="job-meta">{formatSize(job.file.size)} - {job.status}</p>
        </div>
        <div>
          {job.status === 'error' && <button className="icon-button" onClick={() => onRetry(job.id)} title="Retry" type="button"><RotateCcw size={15} /></button>}
          <button className="icon-button" onClick={() => onRemove(job.id)} title={job.status === 'uploading' ? 'Cancel' : 'Remove'} type="button">
            {job.status === 'queued' ? <Trash2 size={15} /> : <X size={15} />}
          </button>
        </div>
      </div>
      {job.status === 'uploading' && (
        <div className="progress-wrap">
          <div className="progress-labels"><span>{progress < 100 ? `Uploading ${progress}%` : 'Processing on server'}</span><span>{job.file.type || 'file'}</span></div>
          <div className="progress"><span style={{ '--progress': `${progress}%` } as CSSProperties} /></div>
        </div>
      )}
      {job.status === 'done' && job.result && (
        <div className="job-result">
          {statusBadge((job.result.overall_confidence || 0) >= 0.7 ? 'EXTRACTED' : 'LOW_CONFIDENCE', job.result.overall_confidence)}
          <span className="job-meta">{resultFields(job.result).length} fields</span>
          {!!job.result.unresolved_count && <span className="job-meta">{job.result.unresolved_count} unresolved</span>}
          {duration && <span className="job-meta">{duration}</span>}
          <button className="link-button" onClick={() => onOpenReview(job.result?.metadata.extraction_id || '')} type="button">Review</button>
        </div>
      )}
      {job.status === 'error' && <div className="job-error">{job.errorMsg}</div>}
    </article>
  );
}

function ReviewPage({
  correctingField,
  highlightedField,
  onCloseCorrection,
  onCorrectField,
  onHighlightField,
  onNavigate,
  onSaveCorrection,
  onSelectField,
  result,
  selectedField,
}: {
  correctingField: string | null;
  highlightedField: string | null;
  onCloseCorrection: () => void;
  onCorrectField: (field: string) => void;
  onHighlightField: (field: string | null) => void;
  onNavigate: (page: Page) => void;
  onSaveCorrection: (field: string) => void;
  onSelectField: (field: string) => void;
  result: ExtractionResult | null;
  selectedField: string | null;
}) {
  if (!result) {
    return (
      <section className="page">
        <div className="empty-state">
          <div className="icon-box large-icon"><Search size={24} /></div>
          <p>No extraction selected.</p>
          <button className="ghost-button" onClick={() => onNavigate('upload')} type="button">Upload a document</button>
        </div>
      </section>
    );
  }

  const fields = resultFields(result);
  const correction = correctingField && result.fields ? result.fields[correctingField] : null;

  return (
    <section className="review-page">
      <header className="review-header">
        <div className="review-heading">
          <h2>{docType(result)}</h2>
          {statusBadge((result.overall_confidence || 0) >= 0.7 ? 'EXTRACTED' : 'LOW_CONFIDENCE', result.overall_confidence)}
          {!!result.unresolved_count && statusBadge('UNRESOLVED')}
        </div>
        <div className="job-meta">{fields.length} fields - {Math.round(result.metadata.processing_time_ms || 0)}ms</div>
      </header>
      <div className="review-body">
        <PreviewPane highlightedField={highlightedField} result={result} />
        <div className="field-pane">
          <div className="field-meta"><span>{result.metadata.extraction_id.slice(0, 8)}</span><span>{(result.metadata.vlm_model || 'local').split('/').pop()}</span></div>
          <div className="field-list">
            {fields.map(([name, field]) => (
              <button
                className={`field-row ${selectedField === name ? 'selected' : ''}`}
                key={name}
                onClick={() => onSelectField(name)}
                onMouseEnter={() => onHighlightField(name)}
                onMouseLeave={() => onHighlightField(null)}
                type="button"
              >
                <span className="field-name">{name}</span>
                <span className={`field-value ${field.value === null || field.value === undefined ? 'empty' : ''}`}>
                  {field.value === null || field.value === undefined ? 'null' : String(field.value)}
                </span>
                {statusBadge(field.status, field.confidence)}
                <span
                  className="icon-button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onCorrectField(name);
                  }}
                  title="Correct field"
                >
                  <Pencil size={14} />
                </span>
              </button>
            ))}
          </div>
          <div className="field-footer">
            <span>{fields.filter(([, field]) => field.status === 'EXTRACTED').length} extracted</span>
            <span>{fields.filter(([, field]) => field.status === 'LOW_CONFIDENCE').length} low confidence</span>
            <span>{fields.filter(([, field]) => field.status === 'UNRESOLVED' || field.status === 'VALIDATION_FAILURE').length} issues</span>
            <span>{fields.filter(([, field]) => field.status === 'CORRECTED').length} corrected</span>
          </div>
        </div>
      </div>
      {correctingField && correction && (
        <CorrectionDialog field={correction} fieldName={correctingField} onClose={onCloseCorrection} onSave={onSaveCorrection} />
      )}
    </section>
  );
}

function PreviewPane({ highlightedField, result }: { highlightedField: string | null; result: ExtractionResult }) {
  const highlighted = highlightedField && result.fields ? result.fields[highlightedField] : null;
  const highlight = highlighted?.source_bbox ? <span className="field-highlight" style={bboxStyle(highlighted.source_bbox)} /> : null;
  const url = result.__previewUrl;

  if (!url) {
    return (
      <div className="preview-pane">
        <div className="empty-state"><div className="icon-box large-icon"><FileText size={24} /></div><p>Preview is available for newly uploaded files.</p></div>
      </div>
    );
  }

  const isPdf = result.__fileType === 'application/pdf' || /\.pdf$/i.test(result.__fileName || '');
  return (
    <div className="preview-pane">
      <div className="preview-stage">
        {isPdf ? <iframe src={url} title="Document preview" /> : <img src={url} alt="Uploaded document preview" />}
        {highlight}
      </div>
    </div>
  );
}

function CorrectionDialog({
  field,
  fieldName,
  onClose,
  onSave,
}: {
  field: FieldResult;
  fieldName: string;
  onClose: () => void;
  onSave: (field: string) => void;
}) {
  const currentValue = field.value === null || field.value === undefined ? '' : String(field.value);
  return (
    <div
      className="modal-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal-panel" role="dialog" aria-modal="true" aria-labelledby="correction-title">
        <div className="modal-header">
          <h3 id="correction-title">Correct Field</h3>
          <button className="icon-button" onClick={onClose} type="button"><X size={16} /></button>
        </div>
        <div className="modal-grid">
          <div><span className="label">Field</span><p className="mono">{fieldName}</p></div>
          <div><span className="label">Original</span><p className="mono">{currentValue || 'null'}</p></div>
          <label><span className="label">Correction</span><input id="corrValue" className="input" defaultValue={currentValue} autoComplete="off" /></label>
          <label><span className="label">Reason</span><textarea id="corrReason" className="textarea" rows={3} placeholder="Optional note" /></label>
        </div>
        <div className="modal-actions">
          <button className="quiet-button" onClick={onClose} type="button">Cancel</button>
          <button className="primary-button" onClick={() => onSave(fieldName)} type="button">Save</button>
        </div>
      </div>
    </div>
  );
}

function HistoryPage({
  allRows,
  filter,
  filteredRows,
  onFilter,
  onOpenReview,
  onSearch,
  search,
}: {
  allRows: ExtractionResult[];
  filter: HistoryFilter;
  filteredRows: ExtractionResult[];
  onFilter: (filter: HistoryFilter) => void;
  onOpenReview: (id: string) => void;
  onSearch: (search: string) => void;
  search: string;
}) {
  return (
    <section className="page">
      <div className="page-header">
        <div>
          <h2 className="page-title">History</h2>
          <p className="page-copy">{allRows.length} extraction{allRows.length === 1 ? '' : 's'} in this session.</p>
        </div>
      </div>
      <div className="filters">
        <input className="search-box" onChange={(event) => onSearch(event.target.value)} placeholder="Search by type or ID" value={search} />
        <div className="segment" role="tablist">
          {(['all', 'extracted', 'unresolved', 'corrected'] as HistoryFilter[]).map((item) => (
            <button className={`segment-button ${filter === item ? 'active' : ''}`} key={item} onClick={() => onFilter(item)} type="button">
              {item === 'unresolved' ? 'Issues' : item}
            </button>
          ))}
        </div>
      </div>
      <div className="panel">
        {filteredRows.length === 0 ? (
          <div className="empty-state"><p>{allRows.length ? 'No history rows match the filter.' : 'No uploads yet.'}</p></div>
        ) : (
          <div className="rows">
            {filteredRows.map((item) => (
              <button className="history-row" key={item.metadata.extraction_id} onClick={() => onOpenReview(item.metadata.extraction_id)} type="button">
                <span className="doc-cell">
                  <span className="icon-box"><FileText size={17} /></span>
                  <span><span className="doc-title">{docType(item)}</span><span className="job-meta">{item.__fileName || item.metadata.extraction_id.slice(0, 8)}</span></span>
                </span>
                <span>{statusBadge((item.overall_confidence || 0) >= 0.7 ? 'EXTRACTED' : 'LOW_CONFIDENCE', item.overall_confidence)}</span>
                <span className="job-meta">{resultFields(item).length} fields</span>
                <span className="job-meta">{Math.round(item.metadata.processing_time_ms || 0)}ms</span>
                <span className="job-meta">{formatDate(item.metadata.timestamp)}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function SettingsPage({
  apiUrl,
  onApiUrlChange,
  onReset,
  onSave,
  saved,
}: {
  apiUrl: string;
  onApiUrlChange: (url: string) => void;
  onReset: () => void;
  onSave: () => void;
  saved: boolean;
}) {
  const models = [
    ['Vision Model', 'Surya OCR 0.17.1', 'Local neural OCR/layout runtime'],
    ['Logic Model', 'Llama-3.1-8B-Instruct-Q4_K_M', 'Local GGUF runtime'],
    ['Embeddings', 'all-MiniLM-L6-v2', 'Local ChromaDB retrieval'],
  ];
  const guardrails = [
    'Next.js App Router with typed React state',
    'No external frontend CDNs or inline event handlers',
    'React-escaped rendering for server and document values',
    'File picker snapshots FileList before clearing the input',
    'Backend file type and size validation before extraction',
  ];

  return (
    <section className="page narrow">
      <div className="page-header">
        <div>
          <h2 className="page-title">Settings</h2>
          <p className="page-copy">Backend endpoint and local runtime details.</p>
        </div>
      </div>
      <div className="settings-stack">
        <div className="panel settings-panel">
          <div className="settings-title"><span className="icon-box"><Server size={17} /></span><div><h3>Backend Connection</h3><p className="subtle">FastAPI extraction endpoint</p></div></div>
          <label><span className="label">API Base URL</span><input className="input mono" onChange={(event) => onApiUrlChange(event.target.value)} value={apiUrl} /></label>
          {saved && <p className="upload-note">Settings saved.</p>}
        </div>
        <div className="panel settings-panel">
          <div className="settings-title"><span className="icon-box"><Cpu size={17} /></span><div><h3>Models</h3><p className="subtle">Local privacy-first inference stack</p></div></div>
          <div className="setting-list">
            {models.map(([label, value, sub]) => (
              <div className="setting-row" key={label}><span><strong>{label}</strong><br /><span className="subtle">{sub}</span></span><span className="mono">{value}</span></div>
            ))}
          </div>
        </div>
        <div className="panel settings-panel">
          <div className="settings-title"><span className="icon-box"><Shield size={17} /></span><div><h3>Frontend Guardrails</h3><p className="subtle">Structural changes that reduce upload regressions</p></div></div>
          <div className="setting-list">{guardrails.map((item) => <div className="setting-row" key={item}><span>{item}</span><Check size={16} /></div>)}</div>
        </div>
        <div>
          <button className="primary-button" onClick={onSave} type="button">Save Settings</button>
          <button className="ghost-button" onClick={onReset} type="button">Reset</button>
        </div>
      </div>
    </section>
  );
}
