package com.example.mobileclassifiertest;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_LOAD_MODEL = 1;
    private static final int REQUEST_LOAD_CLASSES = 2;
    private static final int REQUEST_LOAD_IMAGE = 3;
    private static final int REQUEST_RUN_CAMERA = 4;
    private static final String TAG = "main";

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    private int REQUEST_CODE_PERMISSION = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permissions.CAMERA"};

    private Button btnLoadModel;
    private Button btnLoadClasses;
    private Button btnLoadImage;
    private ImageView imagePreview;
    private TextView tvResult;
    private PreviewView previewView;
    private Button btnRunCamera;
    private List<String> classes;
    private View previewContainer;
    private View imageViewContainer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnLoadModel = findViewById(R.id.btn_load_model);
        btnLoadClasses = findViewById(R.id.btn_load_classes);
        btnLoadImage = findViewById(R.id.btn_load_image);
        btnRunCamera = findViewById(R.id.btn_run_camera);

        imagePreview = findViewById(R.id.image_preview);
        previewView = findViewById(R.id.preview_view);
        tvResult = findViewById(R.id.tv_result);

        previewContainer = findViewById(R.id.preview_container);
        imageViewContainer = findViewById(R.id.imageView_container);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            // Request the permissions if they are not granted.
            ActivityCompat.requestPermissions(this, new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, REQUEST_CODE_PERMISSION);
        } else {
            // Permissions are already granted.
            // Proceed with your app's functionality.
        }



        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed.");
        } else {
            Log.d(TAG, "OpenCV initialization succeeded.");
        }

        btnLoadModel.setOnClickListener(v -> {
            // Logic to load PyTorch model
            // For simplicity, we'll use Intent to load files. You can adjust as needed.
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("*/*");
            startActivityForResult(intent, REQUEST_LOAD_MODEL);
        });

        btnLoadClasses.setOnClickListener(v -> {
            // Logic to load Classes.txt
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("*/*");
            startActivityForResult(intent, REQUEST_LOAD_CLASSES);
        });

        btnLoadImage.setOnClickListener(v -> {
            LoadTorchModule("model.ptl");
            classes = LoadClasses("classes.txt");
            previewContainer.setVisibility(View.INVISIBLE);
            imageViewContainer.setVisibility(View.VISIBLE);
            // Logic to load Image to Classify
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, REQUEST_LOAD_IMAGE);
        });

        btnRunCamera.setOnClickListener(v -> {
            LoadTorchModule("model.ptl");
            classes = LoadClasses("classes.txt");
            previewContainer.setVisibility(View.VISIBLE);
            imageViewContainer.setVisibility(View.INVISIBLE);

            // start camera and run model
            if (checkPermissions()) {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSION);
            }

            cameraProviderFuture = ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(() -> {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    startCamera(cameraProvider);
                } catch (ExecutionException | InterruptedException e) {
                    // errors
                }
            }, ContextCompat.getMainExecutor(this));
        });



//        File file = new File(this.getFilesDir(),"model.ptl");
//        boolean result = file.delete();
//
//        File file2 = new File(this.getFilesDir(),"classes.ptl");
//        boolean result2 = file.delete();

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK && data != null) {
            switch (requestCode) {
                case REQUEST_LOAD_MODEL:
                    Uri modelUri = data.getData();
                    // Save or use the modelUri as needed
                    copyFileToInternalStorage(modelUri, "model.ptl");
                    break;
                case REQUEST_LOAD_CLASSES:
                    Uri classesUri = data.getData();
                    // Save or use the classesUri as needed
                    copyFileToInternalStorage(classesUri, "classes.txt");
                    break;
                case REQUEST_LOAD_IMAGE:
                    try {
                        // Display the loaded image
                        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(data.getData()));
//                        imagePreview.setImageBitmap(bitmap);

                        // After loading the image, you'd typically process it and display the result
                        processImage(bitmap, true);

                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    break;
            }
        }
    }

    private void processImage(Bitmap bitmap, boolean showImg) {





//
        float[] mean_norm = TensorImageUtils.TORCHVISION_NORM_MEAN_RGB;
        float[] std_norm = TensorImageUtils.TORCHVISION_NORM_STD_RGB;

        // Resize Bitmap to 256x256
        Bitmap resizedBitmap = resizeBitmapTo256x256(bitmap);

        // Center Crop to 224x224
        Bitmap croppedBitmap = centerCropTo224x224(resizedBitmap);

        if (showImg) {
            imagePreview.setImageBitmap(croppedBitmap);
        }

        // Convert cropped Bitmap to Tensor
        Tensor inputTensor =  TensorImageUtils.bitmapToFloat32Tensor(croppedBitmap, mean_norm, std_norm);

//        Bitmap inputTensor_bitmap_tmp = tensorToBitmap(inputTensor);
//        imagePreview.setImageBitmap(inputTensor_bitmap_tmp);


        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }
        String classResult = classes.get(maxScoreIdx);
        Log.v("Torch", "Detected " + classResult);

        tvResult.setText("Classification result:" + classResult);
    }



    Module module;
    void LoadTorchModule(String fileName) {

        File modelFile = new File(this.getFilesDir(), fileName);
        String path = modelFile.getAbsolutePath();


        module = LiteModuleLoader.load(path);
    }

    private void copyFileToInternalStorage(Uri uri, String outputFileName) {
        try (InputStream is = getContentResolver().openInputStream(uri);
             OutputStream os = new FileOutputStream(new File(getFilesDir(), outputFileName))) {
            byte[] buffer = new byte[1024];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    List<String> LoadClasses(String fileName){
        List<String> classes = new ArrayList<>();
        try {
            File file = new File(getFilesDir(), fileName);
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
            String line;
            while ((line = br.readLine()) != null){
                classes.add(line);
            }
        }
        catch (IOException e){
            e.printStackTrace();
        }
        return classes;
    }


    private Bitmap resizeBitmapTo256x256(Bitmap source) {
        float scale;
        int newWidth;
        int newHeight;

        if (source.getWidth() > source.getHeight()) {
            // landscape or square
            scale = 256.0f / source.getHeight();
            newWidth = Math.round(source.getWidth() * scale);
            newHeight = 256;
        } else {
            // portrait
            scale = 256.0f / source.getWidth();
            newHeight = Math.round(source.getHeight() * scale);
            newWidth = 256;
        }
        return Bitmap.createScaledBitmap(source, newWidth, newHeight, false);
    }

    private Bitmap centerCropTo224x224(Bitmap source) {
        int x = (source.getWidth() - 224) / 2;
        int y = (source.getHeight() - 224) / 2;
        return Bitmap.createBitmap(source, x, y, 224, 224);
    }

    public Bitmap tensorToBitmap(Tensor inputTensor) {
        // Adjusting for the tensor shape [1, 3, height, width]
        int height = (int) inputTensor.shape()[2];
        int width = (int) inputTensor.shape()[3];

        float[] tensorData = inputTensor.getDataAsFloatArray();

        // Convert tensor data to ARGB byte array for bitmap
        int numPixels = width * height;
        int[] argbPixels = new int[numPixels];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;

                float tmp = tensorData[x + (y * width) + (0 * height * width)];
                if (tmp > 1)
                    tmp = 1;
                if (tmp < 0)
                    tmp = 0;
                int r = (int) (tmp * 255);
                tmp = tensorData[x + (y * width) + (1 * height * width)];
                if (tmp > 1)
                    tmp = 1;
                if (tmp < 0)
                    tmp = 0;
                int g = (int) (tmp * 255);
                tmp = tensorData[x + (y * width) + (2 * height * width)];
                if (tmp > 1)
                    tmp = 1;
                if (tmp < 0)
                    tmp = 0;
                int b = (int) (tmp * 255);
                argbPixels[index] = Color.argb(255, r, g, b); // Alpha is set to 255 (fully opaque)
            }
        }

        // Create bitmap from ARGB pixel array
        Bitmap outputBitmap = Bitmap.createBitmap(argbPixels, width, height, Bitmap.Config.ARGB_8888);

        return outputBitmap;
    }

    private Bitmap rotateBitmap(Bitmap source, int rotation) {
        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }


    private boolean checkPermissions() {
        for (String permission : REQUIRED_PERMISSIONS){
            if (ContextCompat.checkSelfPermission(this,permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }


    Executor executor = Executors.newSingleThreadExecutor();

    void startCamera(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
//                .setTargetResolution(new android.util.Size(256,256))
                .setTargetResolution(new android.util.Size(765, 1020))
//                .setTargetResolution(new android.util.Size(3060,4080))
//                .setTargetResolution(new android.util.Size(224,224))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                int rotation = image.getImageInfo().getRotationDegrees();
                analyzeImage(image, rotation);
                image.close();
            }
        });

        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalysis);
    }

    void analyzeImage(ImageProxy image, int rotation)
    {
        Bitmap bitmap = imageProxyToBitmap(image);
        processImage(bitmap, false);

    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        @SuppressLint("UnsafeOptInUsageError")
        Image.Plane[] planes = image.getImage().getPlanes();

        // Luminance plane (Y)
        ByteBuffer yBuffer = planes[0].getBuffer();
        int ySize = yBuffer.remaining();
        byte[] yBytes = new byte[ySize];
        yBuffer.get(yBytes);

        // U and V planes
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        // Strides
        int yPixelStride = planes[0].getPixelStride();
        int yRowStride = planes[0].getRowStride();
        int uvPixelStride = planes[1].getPixelStride();
        int uvRowStride = planes[1].getRowStride();

        int width = image.getWidth();
        int height = image.getHeight();

        byte[] yuvBytes = new byte[width * height * 3 / 2];

        int yPos = 0;
        for (int row = 0; row < height; row++) {
            int srcRowOffset = yRowStride * row;
            System.arraycopy(yBytes, srcRowOffset, yuvBytes, yPos, width);
            yPos += width;
        }

        byte[] uBytes = new byte[uBuffer.remaining()];
        uBuffer.get(uBytes);

        byte[] vBytes = new byte[vBuffer.remaining()];
        vBuffer.get(vBytes);

        int uvPos = width * height;
        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                yuvBytes[uvPos++] = vBytes[row * uvRowStride + col * uvPixelStride];
                yuvBytes[uvPos++] = uBytes[row * uvRowStride + col * uvPixelStride];
            }
        }

        Mat yuvMat = new Mat(new Size(width, height + height / 2), CvType.CV_8UC1);
        yuvMat.put(0, 0, yuvBytes);

        Mat rgbMat = new Mat();
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21);

        Bitmap bmp = Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgbMat, bmp);

        yuvMat.release();
        rgbMat.release();

        return bmp;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission was granted. Proceed with your app's functionality.
            } else {
                // Permission denied by the user. You may inform the user about why the permission is needed.
            }
        }
    }

}