/* ═══════════════════════════════════════════
   capture.js — Web-Based Face Capture System
   ═══════════════════════════════════════════ */
(function () {
  'use strict';

  /* ── Config ── */
  const CONFIG = {
    targetImages: 50,
    uploadUrl: '/api/upload_capture',
    captureInterval: 1,        // seconds between captures
    imageQuality: 0.92,
    minFaceDetections: 20,
    confidenceThreshold: 0.5,
  };

  let faceApiReady = false;

  /* ── State ── */
  const state = {
    isCapturing: false,
    imageCount: 0,
    images: [],
    username: '', email: '', userId: '',
    department: '', phone: '', role: '', notes: '',
    stream: null,
    animationId: null,
    frameCount: 0,
    lastCaptureTime: 0,
  };

  /* ── UI refs ── */
  const UI = {};
  const ids = [
    ['usernameForm',    'username-form'],
    ['usernameInput',   'capture-username'],
    ['emailInput',      'capture-email'],
    ['userIdInput',     'capture-id'],
    ['departmentInput', 'capture-department'],
    ['phoneInput',      'capture-phone'],
    ['roleInput',       'capture-role'],
    ['notesInput',      'capture-notes'],
    ['proceedBtn',      'proceed-camera-btn'],
    ['captureInterface','capture-interface'],
    ['video',           'camera-feed'],
    ['canvas',          'face-detection-overlay'],
    ['statusText',      'status-text'],
    ['progressBar',     'capture-progress'],
    ['progressText',    'progress-text'],
    ['startBtn',        'start-capture-btn'],
    ['stopBtn',         'stop-capture-btn'],
    ['submitBtn',       'submit-capture-btn'],
    ['capturedCount',   'captured-count'],
    ['remainingCount',  'remaining-count'],
    ['overlayCount',    'overlay-count'],
  ];

  /* ───────────── helpers ───────────── */

  function initUI() {
    ids.forEach(([key, id]) => { UI[key] = document.getElementById(id); });
    if (UI.canvas) UI.ctx = UI.canvas.getContext('2d');
    if (UI.startBtn)  UI.startBtn.disabled  = true;
    if (UI.stopBtn)   UI.stopBtn.disabled   = true;
    if (UI.submitBtn) UI.submitBtn.disabled  = true;
  }

  function bindEvents() {
    UI.proceedBtn?.addEventListener('click', handleProceed);
    UI.startBtn?.addEventListener('click',   startCapture);
    UI.stopBtn?.addEventListener('click',    stopCapture);
    UI.submitBtn?.addEventListener('click',  submitCaptures);
  }

  function showStatus(msg, type = 'info') {
    if (UI.statusText) UI.statusText.textContent = msg;
    const el = document.getElementById('capture-status');
    if (!el) return;
    const ic = { success:'check-circle', danger:'exclamation-circle',
                 warning:'exclamation-triangle', info:'info-circle' };
    el.className = 'status-bar status-' + type;
    el.innerHTML = '<i class="fas fa-' + (ic[type]||'info-circle') + '"></i> ' + msg;
  }

  function setStep(n) {
    [1,2,3].forEach(i => {
      const s = document.getElementById('step-ind-' + i);
      if (s) s.classList.toggle('active', i <= n);
      const l = document.getElementById('step-line-' + (i - 1));
      if (l) l.classList.toggle('done', i <= n);
    });
  }

  /* ───────────── Face-API ───────────── */

  async function loadFaceApi() {
    if (typeof faceapi === 'undefined') { faceApiReady = false; return; }
    try {
      const url = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
      await faceapi.nets.tinyFaceDetector.loadFromUri(url);
      faceApiReady = true;
      showStatus('Face detection ready – capturing images…', 'success');
    } catch {
      faceApiReady = false;
      showStatus('Basic capture mode (no face detection)', 'info');
    }
  }

  /* ───────────── Step 1 → 2 ───────────── */

  async function handleProceed() {
    const v = (id) => document.getElementById(id)?.value.trim() ?? '';
    const username   = v('capture-username');
    const email      = v('capture-email');
    const userId     = v('capture-id');
    const department = document.getElementById('capture-department')?.value ?? '';
    const role       = document.getElementById('capture-role')?.value ?? '';

    if (!username || !/^[A-Za-z\s]+$/.test(username))
      return showStatus('Enter a valid name (letters & spaces only)', 'danger');
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
      return showStatus('Enter a valid email address', 'danger');
    if (!userId) return showStatus('Enter your ID number', 'danger');
    if (!department) return showStatus('Select a department', 'danger');
    if (!role) return showStatus('Select a role', 'danger');

    Object.assign(state, {
      username, email, userId, department,
      phone: v('capture-phone'),
      role,
      notes: v('capture-notes'),
    });

    if (UI.usernameForm)    UI.usernameForm.style.display    = 'none';
    if (UI.captureInterface) UI.captureInterface.style.display = 'block';
    setStep(2);
    showStatus('Requesting camera access…', 'info');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, aspectRatio: { ideal: 16/9 } },
        audio: false,
      });
      state.stream = stream;
      if (UI.video) UI.video.srcObject = stream;
      showStatus('Camera ready — click "Start Capture"', 'success');
      if (UI.startBtn) UI.startBtn.disabled = false;
    } catch (err) {
      showStatus('Camera access denied. Check permissions.', 'danger');
      if (UI.usernameForm)    UI.usernameForm.style.display    = 'block';
      if (UI.captureInterface) UI.captureInterface.style.display = 'none';
      setStep(1);
    }
  }

  /* ───────────── Capture loop ───────────── */

  function startCapture() {
    if (state.isCapturing) return;
    state.isCapturing = true;
    state.imageCount = 0;
    state.images = [];
    state.frameCount = 0;
    state.lastCaptureTime = Date.now();

    if (UI.startBtn) UI.startBtn.disabled = true;
    if (UI.stopBtn)  UI.stopBtn.disabled  = false;
    showStatus('Loading face detection models…', 'info');

    const vc = document.getElementById('video-container');
    if (vc) vc.classList.add('recording');

    loadFaceApi().then(() => {
      showStatus('Capturing — position your face in center', 'info');
      captureLoop();
    });
  }

  function captureLoop() {
    if (!state.isCapturing) return;
    state.frameCount++;
    if (UI.canvas && UI.video && UI.ctx) {
      UI.ctx.drawImage(UI.video, 0, 0, UI.canvas.width, UI.canvas.height);
      detectAndCapture();
    }
    state.animationId = requestAnimationFrame(captureLoop);
  }

  async function detectAndCapture() {
    const now = Date.now();
    const elapsed = now - state.lastCaptureTime;
    try {
      if (faceApiReady && typeof faceapi !== 'undefined') {
        const dets = await faceapi.detectAllFaces(UI.video,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: CONFIG.confidenceThreshold }));
        if (dets?.length) {
          drawFaceBox(dets[0]);
          if (elapsed >= CONFIG.captureInterval * 1000) { captureImage(); state.lastCaptureTime = now; }
        } else {
          drawNoFace();
        }
      } else if (elapsed >= CONFIG.captureInterval * 1000) {
        captureImage(); state.lastCaptureTime = now;
      }
    } catch {
      if (elapsed >= CONFIG.captureInterval * 1000) { captureImage(); state.lastCaptureTime = now; }
    }
  }

  /* ── drawing helpers ── */

  function drawFaceBox(det) {
    if (!UI.ctx || !UI.canvas || !UI.video) return;
    UI.ctx.drawImage(UI.video, 0, 0, UI.canvas.width, UI.canvas.height);
    const box = det.box || det.detection?.box;
    UI.ctx.strokeStyle = '#06d6a0'; UI.ctx.lineWidth = 3;
    UI.ctx.strokeRect(box.x, box.y, box.width, box.height);
    // cross-hair
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    UI.ctx.beginPath();
    UI.ctx.moveTo(cx - 10, cy); UI.ctx.lineTo(cx + 10, cy);
    UI.ctx.moveTo(cx, cy - 10); UI.ctx.lineTo(cx, cy + 10);
    UI.ctx.stroke();
  }

  function drawNoFace() {
    if (!UI.ctx || !UI.canvas || !UI.video) return;
    UI.ctx.drawImage(UI.video, 0, 0, UI.canvas.width, UI.canvas.height);
    const cx = UI.canvas.width / 2, cy = UI.canvas.height / 2;
    UI.ctx.strokeStyle = '#f72585'; UI.ctx.lineWidth = 2;
    UI.ctx.strokeRect(cx - 90, cy - 65, 180, 130);
    UI.ctx.fillStyle = '#f72585'; UI.ctx.font = 'bold 15px sans-serif';
    UI.ctx.textAlign = 'center';
    UI.ctx.fillText('NO FACE DETECTED', cx, cy);
    UI.ctx.font = '13px sans-serif';
    UI.ctx.fillText('Position your face here', cx, cy + 22);
    UI.ctx.textAlign = 'left';
  }

  /* ── image capture ── */

  function captureImage() {
    if (!UI.canvas || state.imageCount >= CONFIG.targetImages) return;
    try {
      state.images.push(UI.canvas.toDataURL('image/jpeg', CONFIG.imageQuality));
      state.imageCount++;
      updateProgress();
      playBeep();
      if (state.imageCount >= CONFIG.targetImages) stopCapture();
    } catch (e) { console.error('Capture error:', e); }
  }

  function updateProgress() {
    const pct = (state.imageCount / CONFIG.targetImages) * 100;
    if (UI.progressBar) { UI.progressBar.style.width = pct + '%'; UI.progressBar.setAttribute('aria-valuenow', state.imageCount); }
    if (UI.progressText) UI.progressText.textContent = state.imageCount + ' / ' + CONFIG.targetImages;
    if (UI.capturedCount) UI.capturedCount.textContent = state.imageCount;
    if (UI.overlayCount) UI.overlayCount.textContent = state.imageCount;
    if (UI.remainingCount) UI.remainingCount.textContent = Math.max(0, CONFIG.targetImages - state.imageCount);
  }

  /* ── stop capture ── */

  function stopCapture() {
    if (!state.isCapturing) return;
    state.isCapturing = false;
    if (state.animationId) cancelAnimationFrame(state.animationId);
    if (UI.ctx && UI.canvas) UI.ctx.clearRect(0, 0, UI.canvas.width, UI.canvas.height);
    const vc = document.getElementById('video-container');
    if (vc) vc.classList.remove('recording');
    if (UI.startBtn) UI.startBtn.disabled = false;
    if (UI.stopBtn)  UI.stopBtn.disabled  = true;
    if (UI.submitBtn) UI.submitBtn.disabled = state.imageCount < CONFIG.minFaceDetections;
    showStatus('Captured ' + state.imageCount + ' images. Click "Submit & Save" to register.', 'success');
  }

  /* ───────────── Submit ───────────── */

  async function submitCaptures() {
    if (state.imageCount < CONFIG.minFaceDetections)
      return showStatus('Need at least ' + CONFIG.minFaceDetections + ' images', 'danger');
    if (!state.username) return showStatus('Username missing', 'danger');

    const ok = await confirmDialog(
      'Ready to save ' + state.imageCount + ' face images for ' + state.username + '?',
      'This will train the recognition model.'
    );
    if (!ok) return showStatus('Submission cancelled.', 'info');

    if (UI.submitBtn) UI.submitBtn.disabled = true;
    showStatus('Uploading & processing…', 'info');

    try {
      const res = await fetch(CONFIG.uploadUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: state.username, email: state.email,
          userId: state.userId, department: state.department,
          phone: state.phone, role: state.role, notes: state.notes,
          images: state.images,
        }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        showStatus('✓ ' + data.message, 'success');
        setStep(3);
        setTimeout(() => showSuccessOverlay(data), 400);
      } else {
        showStatus('Error: ' + (data.error || 'Upload failed'), 'danger');
        if (UI.submitBtn) UI.submitBtn.disabled = false;
      }
    } catch (err) {
      showStatus('Network error: ' + err.message, 'danger');
      if (UI.submitBtn) UI.submitBtn.disabled = false;
    }
  }

  /* ── success overlay ── */

  function showSuccessOverlay() {
    const bg = document.createElement('div');
    bg.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:9999;display:flex;align-items:center;justify-content:center;animation:fadeIn .3s ease';
    const card = document.createElement('div');
    card.style.cssText = 'background:#fff;border-radius:20px;padding:2.5rem;max-width:540px;width:92%;box-shadow:0 30px 70px rgba(0,0,0,.5);animation:scaleIn .4s ease;text-align:center;';
    card.innerHTML = `
      <style>
        @keyframes scaleIn{from{transform:scale(.85);opacity:0}to{transform:scale(1);opacity:1}}
        @keyframes fadeIn{from{opacity:0}to{opacity:1}}
        .ok-circle{width:80px;height:80px;margin:0 auto 1.5rem;border-radius:50%;background:linear-gradient(135deg,#06d6a0,#00b894);display:flex;align-items:center;justify-content:center;font-size:2.5rem;color:#fff;box-shadow:0 8px 24px rgba(6,214,160,.3)}
        .dtl{background:#f8f9fa;border-radius:10px;padding:1rem 1.25rem;margin:1.25rem 0;border-left:4px solid #06d6a0;text-align:left}
        .dr{display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid rgba(0,0,0,.05);font-size:.9rem}
        .dr:last-child{border:none}
        .dl{color:#666;font-weight:600;display:flex;gap:6px;align-items:center}
        .dv{color:#1a1a2e;font-weight:700}
        .btn-go{width:100%;padding:.9rem;background:linear-gradient(135deg,#06d6a0,#00b894);color:#fff;border:none;border-radius:12px;font-size:1rem;font-weight:600;cursor:pointer;margin-top:.75rem;transition:transform .2s}
        .btn-go:hover{transform:translateY(-2px)}
      </style>
      <div class="ok-circle"><i class="fas fa-check"></i></div>
      <h2 style="color:#1a1a2e;font-size:1.6rem;margin-bottom:.3rem">Registration Successful!</h2>
      <p style="color:#666;margin-bottom:0">Face data captured &amp; model trained</p>
      <div class="dtl">
        <div class="dr"><span class="dl"><i class="fas fa-user" style="color:#06d6a0"></i> Name</span><span class="dv">${state.username}</span></div>
        <div class="dr"><span class="dl"><i class="fas fa-envelope" style="color:#06d6a0"></i> Email</span><span class="dv" style="font-size:.85rem">${state.email}</span></div>
        <div class="dr"><span class="dl"><i class="fas fa-fingerprint" style="color:#06d6a0"></i> ID</span><span class="dv">${state.userId}</span></div>
        <div class="dr"><span class="dl"><i class="fas fa-building" style="color:#06d6a0"></i> Dept</span><span class="dv">${state.department}</span></div>
        <div class="dr"><span class="dl"><i class="fas fa-user-tag" style="color:#06d6a0"></i> Role</span><span class="dv">${state.role}</span></div>
        <div class="dr"><span class="dl"><i class="fas fa-images" style="color:#06d6a0"></i> Photos</span><span class="dv">${state.imageCount}</span></div>
      </div>
      <button class="btn-go" onclick="window.location.href='/'"><i class="fas fa-home" style="margin-right:6px"></i>Go to Home</button>
    `;
    bg.appendChild(card);
    document.body.appendChild(bg);
  }

  /* ── confirm dialog ── */

  function confirmDialog(title, msg) {
    return new Promise(resolve => {
      const bg = document.createElement('div');
      bg.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:9998;display:flex;align-items:center;justify-content:center;animation:fadeIn .2s';
      const box = document.createElement('div');
      box.style.cssText = 'background:#fff;border-radius:16px;padding:2rem;max-width:460px;width:90%;box-shadow:0 20px 60px rgba(0,0,0,.4);text-align:center;';
      box.innerHTML = `
        <div style="font-size:2.5rem;margin-bottom:.75rem"><i class="fas fa-check-circle" style="color:#06d6a0"></i></div>
        <h3 style="color:#1a1a2e;margin-bottom:.4rem;font-size:1.1rem">${title}</h3>
        <p style="color:#666;margin-bottom:1.5rem;font-size:.9rem">${msg}</p>
        <div style="display:flex;gap:.75rem;justify-content:center;">
          <button id="cd-cancel" style="padding:.6rem 1.2rem;border:2px solid #ccc;background:#fff;color:#666;border-radius:8px;font-weight:600;cursor:pointer;font-size:.9rem">Cancel</button>
          <button id="cd-retake" style="padding:.6rem 1.2rem;border:2px solid #f72585;background:#fff;color:#f72585;border-radius:8px;font-weight:600;cursor:pointer;font-size:.9rem">Retake</button>
          <button id="cd-submit" style="padding:.6rem 1.2rem;border:none;background:linear-gradient(135deg,#06d6a0,#00b894);color:#fff;border-radius:8px;font-weight:600;cursor:pointer;font-size:.9rem">Submit</button>
        </div>`;
      bg.appendChild(box);
      document.body.appendChild(bg);

      const close = (r) => { bg.remove(); resolve(r); };
      box.querySelector('#cd-cancel').onclick = () => close(false);
      box.querySelector('#cd-retake').onclick = () => {
        close(false);
        state.isCapturing = false; state.imageCount = 0; state.images = [];
        if (UI.submitBtn) UI.submitBtn.disabled = true;
        if (UI.startBtn)  UI.startBtn.disabled  = false;
        updateProgress();
        showStatus('Ready to retake — click "Start Capture"', 'info');
      };
      box.querySelector('#cd-submit').onclick = () => close(true);
    });
  }

  /* ── beep ── */

  function playBeep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator(), g = ctx.createGain();
      osc.connect(g); g.connect(ctx.destination);
      osc.frequency.value = 800; osc.type = 'sine';
      g.gain.setValueAtTime(0.25, ctx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.1);
      osc.start(ctx.currentTime); osc.stop(ctx.currentTime + 0.1);
    } catch { /* audio not available */ }
  }

  /* ───────────── Bootstrap ───────────── */

  document.addEventListener('DOMContentLoaded', () => { initUI(); bindEvents(); });

  window.addEventListener('beforeunload', () => {
    if (state.stream) state.stream.getTracks().forEach(t => t.stop());
    if (state.animationId) cancelAnimationFrame(state.animationId);
  });
})();
