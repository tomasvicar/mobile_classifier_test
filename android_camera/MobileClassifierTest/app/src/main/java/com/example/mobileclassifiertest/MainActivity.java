package com.example.mobileclassifiertest;

import androidx.annotation.NonNull;
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

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import android.content.Intent;
import android.widget.Button;


public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE_SELECT_MODEL = 1;
    private static final int REQUEST_CODE_SELECT_CLASSES = 2;

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    PreviewView previewView;
    TextView textView;
    private int REQUEST_CODE_PERMISSION = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permissions.CAMERA"};
    List<String> classes;
    Button uploadModelButton, uploadClassesButton, startCameraButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.cameraView);
        textView = findViewById(R.id.result_text);

        uploadModelButton = findViewById(R.id.uploadModelButton);
        uploadClassesButton = findViewById(R.id.uploadClassesButton);
        startCameraButton = findViewById(R.id.startCameraButton);

        uploadModelButton.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("*/*"); // This allows any type of file. You can be more specific if needed.
            startActivityForResult(intent, REQUEST_CODE_SELECT_MODEL);
        });

        uploadClassesButton.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("*/*"); // This allows any type of file. You can be more specific if needed.
            startActivityForResult(intent, REQUEST_CODE_SELECT_CLASSES);
        });


        startCameraButton.setOnClickListener(v -> {
            // start camera and run model
            if (checkPermissions()) {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSION);
            }
            classes = LoadClasses("classes.txt");
            LoadTorchModule("model.pt");
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

    void startCamera(@NonNull ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
//                .setTargetResolution(new Size(224,224))
//                .setTargetResolution(new Size(765,1020))
                .setTargetResolution(new Size(3060,4080 ))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                int rotation = image.getImageInfo().getRotationDegrees();
                analyzeImage(image,rotation);
                image.close();
            }
        });



        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalysis);
    }
    Module module;
    void LoadTorchModule(String fileName){
        File modelFile = new File(this.getFilesDir(), fileName);
        try{
            if (!modelFile.exists()){
                InputStream inputStream = getAssets().open(fileName);
                FileOutputStream outputStream = new FileOutputStream(modelFile);
                byte[] buffer = new byte[2048];
                int bytesRead = -1;
                while ((bytesRead = inputStream.read(buffer)) !=-1){
                    outputStream.write(buffer,0,bytesRead);
                }
                inputStream.close();
                outputStream.close();
            }
            module = LiteModuleLoader.load(modelFile.getAbsolutePath());
        }
        catch (IOException e){
            e.printStackTrace();
        }
    }

    void analyzeImage(ImageProxy image, int rotation)
    {
        float[] mean_norm = TensorImageUtils.TORCHVISION_NORM_MEAN_RGB;
        float[] std_norm = TensorImageUtils.TORCHVISION_NORM_STD_RGB;
//        float[] mean_norm = {0.485f, 0.456f, 0.406f};
//        float[] std_norm = {0.229f, 0.224f, 0.225f};


//        @SuppressLint("UnsafeOptInUsageError") Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(Objects.requireNonNull(image.getImage()),
//                rotation, 224, 224,
//                mean_norm, std_norm);

        Bitmap originalBitmap = imageProxyToBitmap(image);

        // Resize Bitmap to 256x256
//        Bitmap resizedBitmap = resizeBitmapTo256x256(originalBitmap);

        // Center Crop to 224x224
//        Bitmap croppedBitmap = centerCropTo224x224(resizedBitmap);

        Bitmap resizedBitmap = resizeBitmapTo224x224(originalBitmap);

        // Convert cropped Bitmap to Tensor
        Tensor inputTensor = bitmapToFloat32Tensor(resizedBitmap, mean_norm, std_norm);



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
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                textView.setText(classResult);
            }
        });

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


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_CODE_SELECT_MODEL) {
                Uri modelUri = data.getData();
                // Save or use the modelUri as needed
                copyFileToInternalStorage(modelUri, "model.pt");
            } else if (requestCode == REQUEST_CODE_SELECT_CLASSES) {
                Uri classesUri = data.getData();
                // Save or use the classesUri as needed
                copyFileToInternalStorage(classesUri, "classes.txt");
            }
        }
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



//    private Bitmap resizeBitmapTo256x256(Bitmap source) {
//        float scale;
//        int newWidth;
//        int newHeight;
//
//        if (source.getWidth() > source.getHeight()) {
//            // landscape or square
//            scale = 256.0f / source.getHeight();
//            newWidth = Math.round(source.getWidth() * scale);
//            newHeight = 256;
//        } else {
//            // portrait
//            scale = 256.0f / source.getWidth();
//            newHeight = Math.round(source.getHeight() * scale);
//            newWidth = 256;
//        }
//        return Bitmap.createScaledBitmap(source, newWidth, newHeight, false);
    }

    private Bitmap resizeBitmapTo224x224(Bitmap source) {
//        float scale;
//        int newWidth;
//        int newHeight;

//        if (source.getWidth() > source.getHeight()) {
//            // landscape or square
//            scale = 224.0f / source.getHeight();
//            newWidth = Math.round(source.getWidth() * scale);
//            newHeight = 224;
//        } else {
//            // portrait
//            scale = 224.0f / source.getWidth();
//            newHeight = Math.round(source.getHeight() * scale);
//            newWidth = 224;
//        }
        return Bitmap.createScaledBitmap(source, 224, 224, true);
    }

//    private Bitmap centerCropTo224x224(Bitmap source) {
//        int x = (source.getWidth() - 224) / 2;
//        int y = (source.getHeight() - 224) / 2;
//        return Bitmap.createBitmap(source, x, y, 224, 224);
//    }

    private Tensor bitmapToFloat32Tensor(Bitmap bitmap, float[] mean, float[] std) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        return TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std);
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        @SuppressLint("UnsafeOptInUsageError")
        Image.Plane[] planes = image.getImage().getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
}
