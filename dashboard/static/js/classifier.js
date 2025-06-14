const dropbox = document.getElementById('dropbox');
const fileInput = document.getElementById('fileInput');
const filePreviews = document.getElementById('filePreviews');
const browseBtn = document.querySelector('.browse-btn');
const submitBtn = document.querySelector('.submit-btn');
const uploadStats = document.getElementById('uploadStats');
const fileCountEl = document.getElementById('fileCount');
const totalSizeEl = document.getElementById('totalSize');
const progressFill = document.getElementById('progressFill');
const emptyState = document.getElementById('emptyState');

// Modal elements
const previewModal = document.getElementById('previewModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const closeModal = document.getElementById('closeModal');
const configurationModal = document.getElementById('configurationModal');
const configCloseModal = document.getElementById('configCloseModal')

let fileCount = 0;
let totalSize = 0;
let uploadedFiles = []; // Store uploaded files data

// Initialize display
updateDisplay();

// Prevent default behavior for drag and drop events
dropbox.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropbox.classList.add('dragover');
});

dropbox.addEventListener('dragleave', () => {
    dropbox.classList.remove('dragover');
});

// Handle file drop
dropbox.addEventListener('drop', (e) => {
    e.preventDefault();
    dropbox.classList.remove('dragover');
    const files = e.dataTransfer.files;
    handleFiles(files);
});

// Handle file selection through the input
fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    handleFiles(files);
});

// Trigger file input when dropbox or browse button is clicked
dropbox.addEventListener('click', () => {
    fileInput.click();
});

browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
});

submitBtn.addEventListener('click', () => {
    openConfigurationModal()
})

// Modal event listeners
closeModal.addEventListener('click', () => {
    previewModal.classList.remove('show');
});

configCloseModal.addEventListener('click', () => {
    configurationModal.classList.remove('show');
});

previewModal.addEventListener('click', (e) => {
    if (e.target === previewModal) {
        previewModal.classList.remove('show');
    }
});

// Handle files
function handleFiles(files) {
    const allowedTypes = [
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'text/csv'
    ];
    const maxSize = 200 * 1024 * 1024; // 200 MB
    
    if (files.length > 0) {
        let validFiles = 0;
        let invalidFiles = 0;
        
        Array.from(files).forEach(file => {
            if (!allowedTypes.includes(file.type)) {
                showNotification(`${file.name} is not a valid file type.`, 'error');
                invalidFiles++;
                return;
            }
            
            if (file.size > maxSize) {
                showNotification(`${file.name} is too large. Maximum file size is 200 MB.`, 'error');
                invalidFiles++;
                return;
            }
            
            // Handle valid file
            displayFilePreview(file);
            validFiles++;
        });
        
        if (validFiles > 0) {
            simulateUpload(validFiles);
            showNotification(`${validFiles} file${validFiles !== 1 ? 's' : ''} added successfully.`, 'success');
        }
        
        if (invalidFiles > 0) {
            showNotification(`${invalidFiles} file${invalidFiles !== 1 ? 's' : ''} not added due to errors.`, 'error');
        }
    }
}

function displayFilePreview(file) {
    const fileId = `file-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    // Store file data for later preview
    uploadedFiles.push({
        id: fileId,
        file: file,
        name: file.name,
        size: file.size
    });
    
    const preview = document.createElement('div');
    preview.classList.add('file-preview');
    preview.id = fileId;
    
    const fileSize = formatFileSize(file.size);
    const isCsv = file.type === 'text/csv' || file.name.endsWith('.csv');
    
    let previewContent;
    if (isCsv) {
        previewContent = `
            <div class="file-icon">
                <svg height="64" viewBox="0 0 56 64" width="56" xmlns="http://www.w3.org/2000/svg">
                    <path clip-rule="evenodd" d="m5.106 0c-2.802 0-5.073 2.272-5.073 5.074v53.841c0 2.803 2.271 5.074 5.073 5.074h45.774c2.801 0 5.074-2.271 5.074-5.074v-38.605l-18.903-20.31h-31.945z" fill="#45b058" fill-rule="evenodd"/>
                    <path d="m20.306 43.197c.126.144.198.324.198.522 0 .378-.306.72-.703.72-.18 0-.378-.072-.504-.234-.702-.846-1.891-1.387-3.007-1.387-2.629 0-4.627 2.017-4.627 4.88 0 2.845 1.999 4.879 4.627 4.879 1.134 0 2.25-.486 3.007-1.369.125-.144.324-.233.504-.233.415 0 .703.359.703.738 0 .18-.072.36-.198.504-.937.972-2.215 1.693-4.015 1.693-3.457 0-6.176-2.521-6.176-6.212s2.719-6.212 6.176-6.212c1.8.001 3.096.721 4.015 1.711zm6.802 10.714c-1.782 0-3.187-.594-4.213-1.495-.162-.144-.234-.342-.234-.54 0-.361.27-.757.702-.757.144 0 .306.036.432.144.828.739 1.98 1.314 3.367 1.314 2.143 0 2.827-1.152 2.827-2.071 0-3.097-7.112-1.386-7.112-5.672 0-1.98 1.764-3.331 4.123-3.331 1.548 0 2.881.467 3.853 1.278.162.144.252.342.252.54 0 .36-.306.72-.703.72-.144 0-.306-.054-.432-.162-.882-.72-1.98-1.044-3.079-1.044-1.44 0-2.467.774-2.467 1.909 0 2.701 7.112 1.152 7.112 5.636.001 1.748-1.187 3.531-4.428 3.531zm16.994-11.254-4.159 10.335c-.198.486-.685.81-1.188.81h-.036c-.522 0-1.008-.324-1.207-.81l-4.142-10.335c-.036-.09-.054-.18-.054-.288 0-.36.323-.793.81-.793.306 0 .594.18.72.486l3.889 9.992 3.889-9.992c.108-.288.396-.486.72-.486.468 0 .81.378.81.793.001.09-.017.198-.052.288z" fill="#fff"/>
                </svg>
            </div>
        `;
    } else {
        previewContent = `
            <div class="file-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="48px" height="48px">
                    <path fill="#169154" d="M29,6H15.744C14.781,6,14,6.781,14,7.744v7.259h15V6z"/>
                    <path fill="#18482a" d="M14,33.054v7.202C14,41.219,14.781,42,15.743,42H29v-8.946H14z"/>
                    <path fill="#0c8045" d="M14 15.003H29V24.005000000000003H14z"/>
                    <path fill="#17472a" d="M14 24.005H29V33.055H14z"/>
                    <g><path fill="#29c27f" d="M42.256,6H29v9.003h15V7.744C44,6.781,43.219,6,42.256,6z"/>
                    <path fill="#27663f" d="M29,33.054V42h13.257C43.219,42,44,41.219,44,40.257v-7.202H29z"/>
                    <path fill="#19ac65" d="M29 15.003H44V24.005000000000003H29z"/>
                    <path fill="#129652" d="M29 24.005H44V33.055H29z"/></g>
                    <path fill="#0c7238" d="M22.319,34H5.681C4.753,34,4,33.247,4,32.319V15.681C4,14.753,4.753,14,5.681,14h16.638 C23.247,14,24,14.753,24,15.681v16.638C24,33.247,23.247,34,22.319,34z"/>
                    <path fill="#fff" d="M9.807 19L12.193 19 14.129 22.754 16.175 19 18.404 19 15.333 24 18.474 29 16.123 29 14.013 25.07 11.912 29 9.526 29 12.719 23.982z"/>
                </svg>
            </div>
        `;
    }
    
    preview.innerHTML = `
        <div class="preview-img-container">
            ${previewContent}
        </div>
        <div class="file-info">
            <div class="file-name">${file.name}</div>
            <div class="file-size">${fileSize}</div>
            <div class="file-actions">
                <button class="remove-btn">Remove</button>
                <button class="preview" data-file-id="${fileId}">Preview</button>
            </div>
        </div>
    `;
    
    // Add remove functionality
    const removeBtn = preview.querySelector('.remove-btn');
    removeBtn.addEventListener('click', () => {
        removeFile(fileId, file.size);
    });

    // Add preview functionality
    const previewBtn = preview.querySelector('.preview');
    previewBtn.addEventListener('click', () => {
        openPreviewModal(fileId);
    });
    
    filePreviews.appendChild(preview);
    
    // Update statistics
    fileCount++;
    totalSize += file.size;
    updateDisplay();
}

function openPreviewModal(fileId) {
    const fileData = uploadedFiles.find(f => f.id === fileId);
    if (!fileData) return;

    modalTitle.textContent = `Dataset Preview - ${fileData.name}`;
    modalBody.innerHTML = `
        <div class="loading-spinner">
            <div class="spinner"></div>
        </div>
    `;
    previewModal.classList.add('show');

    // Create FormData to send file to server
    const formData = new FormData();
    formData.append('file', fileData.file);

    // Send file to Flask backend for preview
    fetch('/preview_dataset', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayDatasetPreview(data.preview, data.info);
        } else {
            displayError(data.error || 'Failed to load preview');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        displayError('Failed to load dataset preview');
    });
}

function displayDatasetPreview(previewData, info) {
    const { columns, rows } = previewData;
    
    let tableHTML = `
        <div class="dataset-info">
            <h4>Dataset Information</h4>
            <p><strong>Rows shown:</strong> ${rows.length} of ${info.total_rows}</p>
            <p><strong>Total columns:</strong> ${columns.length}</p>
            <p><strong>File size:</strong> ${info.file_size}</p>
        </div>
        <table class="preview-table">
            <thead>
                <tr>
    `;
    
    columns.forEach(column => {
        tableHTML += `<th>${escapeHtml(column)}</th>`;
    });
    
    tableHTML += `
                </tr>
            </thead>
            <tbody>
    `;
    
    rows.forEach(row => {
        tableHTML += '<tr>';
        columns.forEach(column => {
            const cellValue = row[column] !== undefined && row[column] !== null ? row[column] : '';
            tableHTML += `<td>${escapeHtml(String(cellValue))}</td>`;
        });
        tableHTML += '</tr>';
    });
    
    tableHTML += `
            </tbody>
        </table>
    `;
    
    modalBody.innerHTML = tableHTML;
}

function displayError(message) {
    modalBody.innerHTML = `
        <div class="error-message">
            <strong>Error:</strong> ${escapeHtml(message)}
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function removeFile(fileId, size) {
    const fileElement = document.getElementById(fileId);
    if (fileElement) {
        fileElement.style.opacity = '0';
        fileElement.style.transform = 'scale(0.9)';
        
        setTimeout(() => {
            fileElement.remove();
            
            // Remove from uploadedFiles array
            uploadedFiles = uploadedFiles.filter(f => f.id !== fileId);
            
            // Update statistics
            fileCount--;
            totalSize -= size;
            updateDisplay();
            
            showNotification('File removed', 'success');
        }, 300);
    }
    
    resetFileInput();
}

function updateDisplay() {
    fileCountEl.textContent = fileCount;
    totalSizeEl.textContent = formatFileSize(totalSize);
    
    if (fileCount > 0) {
        uploadStats.style.display = 'block';
        emptyState.style.display = 'none';
        progressFill.style.width = '100%';
        submitBtn.disabled = 0;
    } else {
        uploadStats.style.display = 'none';
        emptyState.style.display = 'block';
        submitBtn.disabled = 1;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showNotification(message, type = 'success') {
    // Simple notification - you can enhance this
    console.log(`${type.toUpperCase()}: ${message}`);
}

function simulateUpload(fileCount) {
    progressFill.style.width = '0%';
    
    setTimeout(() => {
        progressFill.style.width = '30%';
        
        setTimeout(() => {
            progressFill.style.width = '60%';
            
            setTimeout(() => {
                progressFill.style.width = '100%';
            }, 200);
        }, 200);
    }, 100);
}

function resetFileInput() {
    fileInput.value = '';
}

function openConfigurationModal() {
    const modal = configurationModal.querySelector(".modal");
    const modalForm = modal.querySelector("form");
    modalForm.innerHTML = ``;
    i=0
    uploadedFiles.forEach(file => {
        i+=1
        modalForm.innerHTML += `
            <p class="title">${file.name}</p>
            <input type="text" placeholder="Sheet*" name="sheet-${i}" id="file_sheet_${i}" class="input-sheet">
            <input type="text" placeholder="reviews column*" name="col-${i}" id="file_col_${i}" class="input-col">
            <select name="models-${i}" id="model-select-${i}">
                <option value="1">Word Tokenized LSTM</option>
                <option value="2">Subword Tokenized LSTM</option>
                <option value="3">GPT2 Tokenized LSTM</option>
                <option value="4">Word Tokenized GRU</option>
                <option value="5">Subword Tokenized GRU</option>
                <option value="6">GPT2 Tokenized GRU</option>
                <option value="7">Subword Tokenized Transformer</option>
                <option value="8">GPT2 Tokenized Transformer</option>
            </select>
        `;
    });

    modalForm.innerHTML += `
            <button type='submit' id="go-btn">Go</button>
    `;
    configurationModal.classList.add("show");
    const goBtn = modalForm.querySelector("#go-btn");
    goBtn.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Collect form data
        const formData = new FormData();
        console.log(formData)
        let isValid = true;
        let validationErrors = [];
        
        // For each uploaded file, validate and collect configuration
        uploadedFiles.forEach((fileData, index) => {
            const i = index + 1;
            const sheet = document.getElementById(`file_sheet_${i}`).value.trim();
            const column = document.getElementById(`file_col_${i}`).value.trim();
            const model = document.getElementById(`model-select-${i}`).value;
            
            // Basic validation
            if (!sheet || !column) {
                isValid = false;
                validationErrors.push(`File ${fileData.name}: Sheet and column are required`);
                return;
            }
            
            // Add file and its configuration to FormData
            formData.append(`file_${i}`, fileData.file);
            formData.append(`sheet_${i}`, sheet);
            formData.append(`column_${i}`, column);
            formData.append(`model_${i}`, model);
        });
        
        formData.append('file_count', uploadedFiles.length);
        
        if (!isValid) {
            alert('Validation errors:\n' + validationErrors.join('\n'));
            return;
        }
        
        // Show loading state
        goBtn.textContent = 'Processing...';
        goBtn.disabled = true;
        
        // Send to backend
        fetch('/statistics', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Close modal and show results
                configurationModal.classList.remove('show');
                displayStatistics(data.results);
            } else {
                console.log(data.error)
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to process files');
        })
        .finally(() => {
            goBtn.textContent = 'Go';
            goBtn.disabled = false;
        });
    });
}

function displayStatistics(results) {
    // Clear existing content and show statistics
    const contentSection = document.querySelector('.content');
    
    let statisticsHTML = '<div class="statistics-container">';
    
    results.forEach((result, index) => {
        if (result.success) {
            statisticsHTML += `
                <div class="charts-container">
                    <h3>${result.filename}</h3>
                    ${result.charts ? result.charts : ''}
                </div>
            `;
        } else {
            statisticsHTML += `
                <div class="file-error">
                    <h3>${result.filename}</h3>
                    <p class="error">Error: ${result.error}</p>
                </div>
            `;
        }
    });
    
    statisticsHTML += '</div>';
    
    // Replace the import section with statistics
    contentSection.innerHTML = statisticsHTML;
}

function initUI() {
    if (fileCount === 0) {
        uploadStats.style.display = 'none';
        emptyState.style.display = 'block';
    }
}

// Initialize UI
initUI();





