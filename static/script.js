const input = document.getElementById('imageInput');
const fileMeta = document.getElementById('fileMeta');
const dropzone = document.getElementById('dropzone');
const form = document.getElementById('uploadForm');

if (input && fileMeta) {
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    fileMeta.textContent = file
      ? `${file.name} • ${(file.size / 1024).toFixed(1)} KB`
      : 'No file selected';
  });
}

if (dropzone && input) {
  ['dragenter', 'dragover'].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, e => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files && files.length) {
      input.files = files;
      const file = files[0];
      fileMeta.textContent = `${file.name} • ${(file.size / 1024).toFixed(1)} KB`;
    }
  });
}

if (form) {
  form.addEventListener('submit', () => {
    const btn = form.querySelector('button[type="submit"]');
    if (btn) {
      btn.textContent = 'Running...';
      btn.disabled = true;
    }
  });
}