document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const submitBtn = document.getElementById('submit-btn');
    const uploadForm = document.getElementById('upload-form');
    const resultsContainer = document.getElementById('results-container');
    const loadingOverlay = document.querySelector('.loading-overlay');

    let files = [];

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('highlight');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('highlight');
        });
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const droppedFiles = [...e.dataTransfer.files];
        handleFiles(droppedFiles);
    });

    // Handle clicked files
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles([...e.target.files]);
    });

    function handleFiles(newFiles) {
        files = [...files, ...newFiles];
        updateFileList();
        submitBtn.disabled = files.length === 0;
    }

    function updateFileList() {
        fileList.innerHTML = '';
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span><i class="fas fa-file-image"></i>${file.name}</span>
                <button type="button" class="btn btn-link text-danger p-0" data-index="${index}">
                    <i class="fas fa-times"></i>
                </button>
            `;
            fileList.appendChild(fileItem);
        });

        // Add remove file handlers
        document.querySelectorAll('.file-item button').forEach(button => {
            button.addEventListener('click', (e) => {
                const index = parseInt(e.currentTarget.dataset.index);
                files.splice(index, 1);
                updateFileList();
                submitBtn.disabled = files.length === 0;
            });
        });
    }

    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        loadingOverlay.classList.remove('d-none');

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files[]', file);
        });

        try {
            const response = await fetch(`${apiPrefix}/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayResults(data);
                // Clear form
                files = [];
                updateFileList();
                submitBtn.disabled = true;
            } else {
                throw new Error(data.error || 'خطا در پردازش تصاویر');
            }
        } catch (error) {
            displayError(error.message);
        } finally {
            loadingOverlay.classList.add('d-none');
        }
    });

    function displayResults(data) {
        resultsContainer.innerHTML = '';
        
        // Add success message
        const successMessage = document.createElement('div');
        successMessage.className = 'success-message';
        successMessage.innerHTML = 'فایل‌ها با موفقیت پردازش شدند!';
        resultsContainer.appendChild(successMessage);

        // Display results for each image
        data.predictions.forEach((prediction, index) => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            resultCard.innerHTML = `
                <h3 class="mb-4">تصویر شماره ${index + 1}</h3>
                <div class="row">
                    <div class="col-md-6">
                        <img src="${URL.createObjectURL(files[index])}" 
                             alt="تصویر ${index + 1}" 
                             class="img-fluid mb-3">
                    </div>
                    <div class="col-md-6">
                        <h4>متن تشخیص داده شده:</h4>
                        <div class="result-text">${prediction.text || 'متنی شناسایی نشد'}</div>
                    </div>
                </div>
            `;
            resultsContainer.appendChild(resultCard);
        });
    }

    function displayError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        resultsContainer.innerHTML = '';
        resultsContainer.appendChild(errorDiv);
    }
});
