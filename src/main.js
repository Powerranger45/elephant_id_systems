const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs-extra');
const { spawn } = require('child_process');
const express = require('express');
const multer = require('multer');
const cors = require('cors');

class ElephantIDApp {
    constructor() {
        this.mainWindow = null;
        this.pythonProcess = null;
        this.expressApp = null;
        this.server = null;
        this.port = 3001;
        this.isDev = process.argv.includes('--dev');
    }

    async createWindow() {
        // Create the browser window with graphics fixes
        this.mainWindow = new BrowserWindow({
            width: 1400,
            height: 900,
            minWidth: 1200,
            minHeight: 800,
            webPreferences: {
                nodeIntegration: true,
                contextIsolation: false,
                enableRemoteModule: true,
                webSecurity: false
            },
            title: 'Elephant ID System',
            show: true, // Show immediately
            // Add these to fix graphics issues
            webSecurity: false,
            allowRunningInsecureContent: true
        });

        // Disable GPU acceleration to fix graphics issues
        this.mainWindow.webContents.on('dom-ready', () => {
            this.mainWindow.webContents.executeJavaScript(`
                console.log('DOM ready, Electron window loaded successfully');
            `);
        });

        // Setup Express server for Python communication
        await this.setupExpressServer();

        // Start Python backend
        await this.startPythonBackend();

        // Load the app with retry logic
        const startUrl = `http://localhost:${this.port}`;

        try {
            await this.mainWindow.loadURL(startUrl);
            console.log('Successfully loaded URL:', startUrl);
        } catch (error) {
            console.error('Failed to load URL:', error);
            // Retry after a short delay
            setTimeout(async () => {
                try {
                    await this.mainWindow.loadURL(startUrl);
                } catch (retryError) {
                    console.error('Retry failed:', retryError);
                }
            }, 2000);
        }

        if (this.isDev) {
            this.mainWindow.webContents.openDevTools();
        }

        // Handle window closed
        this.mainWindow.on('closed', () => {
            this.mainWindow = null;
            this.cleanup();
        });

        // Prevent external links from opening in app
        this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
            shell.openExternal(url);
            return { action: 'deny' };
        });
    }

    async setupExpressServer() {
        this.expressApp = express();

        // Middleware
        this.expressApp.use(cors());
        this.expressApp.use(express.json({ limit: '50mb' }));
        this.expressApp.use(express.urlencoded({ extended: true, limit: '50mb' }));

        // Serve static files
        this.expressApp.use(express.static(path.join(__dirname, 'frontend')));

        // Configure multer for file uploads with increased limits
        const storage = multer.diskStorage({
            destination: (req, file, cb) => {
                const uploadDir = path.join(app.getPath('temp'), 'elephant-uploads');
                fs.ensureDirSync(uploadDir);
                cb(null, uploadDir);
            },
            filename: (req, file, cb) => {
                cb(null, `${Date.now()}-${file.originalname}`);
            }
        });

        const upload = multer({
            storage: storage,
            limits: {
                fileSize: 2 * 1024 * 1024 * 1024, // 2GB per file
                fieldSize: 100 * 1024 * 1024 // 100MB field size
            }
        });

        // API Routes
        this.setupAPIRoutes(upload);

        // Start server
        return new Promise((resolve) => {
            this.server = this.expressApp.listen(this.port, 'localhost', () => {
                console.log(`Express server running on port ${this.port}`);
                resolve();
            });
        });
    }

    setupAPIRoutes(upload) {
        // Single image processing
        this.expressApp.post('/api/process-single', upload.single('image'), async (req, res) => {
            try {
                if (!req.file) {
                    return res.status(400).json({ error: 'No image file provided' });
                }

                const result = await this.callPythonScript('process_single.py', [req.file.path]);

                // Clean up uploaded file
                fs.remove(req.file.path).catch(console.error);

                res.json(result);
            } catch (error) {
                console.error('Single processing error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Batch processing (now uses folder instead of ZIP)
        this.expressApp.post('/api/process-batch', async (req, res) => {
            try {
                const folderPath = req.body.folder_path;
                const similarityThreshold = req.body.similarity_threshold || 0.85;

                if (!folderPath) {
                    return res.status(400).json({ error: 'No folder path provided' });
                }

                const result = await this.callPythonScript('process_batch.py', [
                    folderPath,
                    similarityThreshold.toString()
                ]);

                res.json(result);
            } catch (error) {
                console.error('Batch processing error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Compare with dataset
        this.expressApp.post('/api/compare-dataset', upload.single('image'), async (req, res) => {
            try {
                if (!req.file) {
                    return res.status(400).json({ error: 'No image file provided' });
                }

                const result = await this.callPythonScript('compare_dataset.py', [req.file.path]);

                // Clean up uploaded file
                fs.remove(req.file.path).catch(console.error);

                res.json(result);
            } catch (error) {
                console.error('Dataset comparison error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Get model info
        this.expressApp.get('/api/model-info', async (req, res) => {
            try {
                const result = await this.callPythonScript('get_model_info.py', []);
                res.json(result);
            } catch (error) {
                console.error('Model info error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Download processed results
        this.expressApp.get('/api/download/:filename', (req, res) => {
            try {
                const filename = req.params.filename;
                const filePath = path.join(app.getPath('temp'), filename);

                if (fs.existsSync(filePath)) {
                    res.download(filePath, (err) => {
                        if (!err) {
                            // Clean up file after download
                            setTimeout(() => {
                                fs.remove(filePath).catch(console.error);
                            }, 5000);
                        }
                    });
                } else {
                    res.status(404).json({ error: 'File not found' });
                }
            } catch (error) {
                console.error('Download error:', error);
                res.status(500).json({ error: error.message });
            }
        });

        // Health check
        this.expressApp.get('/api/health', (req, res) => {
            res.json({ status: 'ok', timestamp: new Date().toISOString() });
        });

        // Serve main page
        this.expressApp.get('/', (req, res) => {
            res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
        });
    }

    async startPythonBackend() {
        const pythonPath = this.getPythonPath();
        const backendPath = this.getPythonBackendPath();

        console.log('Starting Python backend...');
        console.log('Python path:', pythonPath);
        console.log('Backend path:', backendPath);

        // Ensure Python backend exists
        if (!fs.existsSync(backendPath)) {
            throw new Error(`Python backend not found at: ${backendPath}`);
        }

        // Set environment variables
        const env = {
            ...process.env,
            PYTHONPATH: backendPath,
            ELECTRON_MODE: '1'
        };

        this.pythonProcess = spawn(pythonPath, ['-u', path.join(backendPath, 'backend_server.py')], {
            env: env,
            cwd: backendPath
        });

        this.pythonProcess.stdout.on('data', (data) => {
            console.log('Python stdout:', data.toString());
        });

        this.pythonProcess.stderr.on('data', (data) => {
            console.error('Python stderr:', data.toString());
        });

        this.pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
        });

        // Wait a bit for Python to start
        await new Promise(resolve => setTimeout(resolve, 3000));
    }

    async callPythonScript(scriptName, args = []) {
        return new Promise((resolve, reject) => {
            const pythonPath = this.getPythonPath();
            const scriptPath = path.join(this.getPythonBackendPath(), scriptName);

            const pythonProcess = spawn(pythonPath, [scriptPath, ...args], {
                cwd: this.getPythonBackendPath()
            });

            let stdout = '';
            let stderr = '';

            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout);
                        resolve(result);
                    } catch (parseError) {
                        reject(new Error(`Failed to parse Python output: ${parseError.message}`));
                    }
                } else {
                    reject(new Error(`Python script failed with code ${code}: ${stderr}`));
                }
            });

            pythonProcess.on('error', (error) => {
                reject(new Error(`Failed to start Python script: ${error.message}`));
            });
        });
    }

    getPythonPath() {
        if (this.isDev) {
            // In development, use venv Python
            if (process.platform === 'win32') {
                return path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
            } else {
                return path.join(__dirname, '..', 'venv', 'bin', 'python3');
            }
        } else {
            // In production, use bundled Python
            const resourcesPath = process.resourcesPath;
            if (process.platform === 'win32') {
                return path.join(resourcesPath, 'python-backend', 'python', 'python.exe');
            } else {
                return path.join(resourcesPath, 'python-backend', 'python', 'bin', 'python3');
            }
        }
    }

    getPythonBackendPath() {
        if (this.isDev) {
            return path.join(__dirname, '..', 'python-backend');
        } else {
            return path.join(process.resourcesPath, 'python-backend');
        }
    }

    cleanup() {
        if (this.pythonProcess) {
            this.pythonProcess.kill();
            this.pythonProcess = null;
        }

        if (this.server) {
            this.server.close();
            this.server = null;
        }
    }
}

// App event handlers
const elephantApp = new ElephantIDApp();

// Disable GPU acceleration to fix graphics issues
app.disableHardwareAcceleration();

// Add command line switches for better compatibility
app.commandLine.appendSwitch('--disable-gpu');
app.commandLine.appendSwitch('--disable-software-rasterizer');
app.commandLine.appendSwitch('--disable-gpu-sandbox');

app.whenReady().then(() => {
    elephantApp.createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            elephantApp.createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    elephantApp.cleanup();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    elephantApp.cleanup();
});

// IPC handlers for file dialogs and system integration
ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(elephantApp.mainWindow, options);
    return result;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(elephantApp.mainWindow, options);
    return result;
});

ipcMain.handle('select-folder', async (event) => {
    const result = await dialog.showOpenDialog(elephantApp.mainWindow, {
        properties: ['openDirectory'],
        title: 'Select folder containing elephant images'
    });
    return result;
});

ipcMain.handle('get-app-path', async (event, name) => {
    return app.getPath(name);
});

ipcMain.handle('show-item-in-folder', async (event, filePath) => {
    shell.showItemInFolder(filePath);
});

// Handle protocol for development
if (elephantApp.isDev) {
    try {
        require('electron-reload')(__dirname, {
            electron: path.join(__dirname, '..', 'node_modules', '.bin', 'electron'),
            hardResetMethod: 'exit'
        });
    } catch (e) {
        console.log('electron-reload not available, continuing without hot reload');
    }
}
