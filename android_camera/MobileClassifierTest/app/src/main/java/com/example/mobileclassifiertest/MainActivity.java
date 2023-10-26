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
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
//import android.util.Size;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
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


import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;
import android.media.Image;
import java.nio.ByteBuffer;
import android.graphics.Bitmap;



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

    ImageView processedImageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        OpenCVLoader.initDebug();
        setContentView(R.layout.activity_main);

        textView = findViewById(R.id.result_text);

        previewView = findViewById(R.id.cameraView);
        processedImageView = findViewById(R.id.processedImageView);

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
//                .setTargetResolution(new android.util.Size(224,224))
                .setTargetResolution(new android.util.Size(765,1020))
//                .setTargetResolution(new android.util.Size(3060,4080))
//                .setTargetResolution(new android.util.Size(224,224))
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


//        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, null, imageAnalysis);
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

//        float[] mean_norm = TensorImageUtils.TORCHVISION_NORM_MEAN_RGB;
//        float[] std_norm = TensorImageUtils.TORCHVISION_NORM_STD_RGB;

        Bitmap originalBitmap = imageProxyToBitmap(image);

        Bitmap rotatedBitmap = rotateBitmap(originalBitmap, rotation);

        Bitmap resizedBitmap = resizeBitmapTo224x224(rotatedBitmap);



        // Convert cropped Bitmap to Tensor
        Tensor inputTensor = bitmapToFloat32Tensor(resizedBitmap, mean_norm, std_norm);

        Bitmap inputTensor_bitmap = tensorToBitmap(inputTensor);

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
                processedImageView.setImageBitmap(inputTensor_bitmap);
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




    private Bitmap resizeBitmapTo224x224(Bitmap source) {
        return Bitmap.createScaledBitmap(source, 224, 224, true);
    }


    private Tensor bitmapToFloat32Tensor(Bitmap bitmap, float[] mean, float[] std) {
//        int width = bitmap.getWidth();
//        int height = bitmap.getHeight();
//        int[] pixels = new int[width * height];
//        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        return TensorImageUtils.bitmapToFloat32Tensor(bitmap, mean, std);
    }

    private Bitmap rotateBitmap(Bitmap source, int rotation) {
        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

//    private Bitmap imageProxyToBitmap(ImageProxy image) {
//        @SuppressLint("UnsafeOptInUsageError")
//        Image.Plane[] planes = image.getImage().getPlanes();
//        ByteBuffer yBuffer = planes[0].getBuffer();
//        ByteBuffer uBuffer = planes[1].getBuffer();
//        ByteBuffer vBuffer = planes[2].getBuffer();
//
//        int ySize = yBuffer.remaining();
//        int uSize = uBuffer.remaining();
//        int vSize = vBuffer.remaining();
//
//        byte[] nv21 = new byte[ySize + uSize + vSize];
//        //U and V are swapped
//        yBuffer.get(nv21, 0, ySize);
//        vBuffer.get(nv21, ySize, vSize);
//        uBuffer.get(nv21, ySize + vSize, uSize);
//
//        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
//        ByteArrayOutputStream out = new ByteArrayOutputStream();
//        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);
//        byte[] imageBytes = out.toByteArray();
//        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
//    }


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






//    private Bitmap imageProxyToBitmap(ImageProxy image) {
//        @SuppressLint("UnsafeOptInUsageError")
//        Image.Plane[] planes = image.getImage().getPlanes();
//        ByteBuffer yBuffer = planes[0].getBuffer();
//        ByteBuffer uBuffer = planes[1].getBuffer();
//        ByteBuffer vBuffer = planes[2].getBuffer();
//
//        int ySize = yBuffer.remaining();
//        int uSize = uBuffer.remaining();
//        int vSize = vBuffer.remaining();
//
//        byte[] yuvBytes = new byte[ySize + uSize + vSize];
//
//        yBuffer.get(yuvBytes, 0, ySize);
//        vBuffer.get(yuvBytes, ySize, vSize);
//        uBuffer.get(yuvBytes, ySize + vSize, uSize);
//
//        Mat yuvMat = new Mat(new Size(image.getWidth(), image.getHeight() + image.getHeight() / 2), CvType.CV_8UC1);
//        yuvMat.put(0, 0, yuvBytes);
//
//        Mat rgbMat = new Mat();
//        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV12);
//
//        Bitmap bmp = Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(rgbMat, bmp);
//
//        return bmp;
//    }

//    private Bitmap imageProxyToBitmap(ImageProxy image) {
//        @SuppressLint("UnsafeOptInUsageError")
//        Image.Plane[] planes = image.getImage().getPlanes();
//        ByteBuffer yBuffer = planes[0].getBuffer();
//        ByteBuffer uBuffer = planes[1].getBuffer();
//        ByteBuffer vBuffer = planes[2].getBuffer();
//
//        int ySize = yBuffer.remaining();
//        int uSize = uBuffer.remaining();
//        int vSize = vBuffer.remaining();
//
//        byte[] nv21 = new byte[ySize + uSize + vSize];
//        yBuffer.get(nv21, 0, ySize);
//        vBuffer.get(nv21, ySize, vSize);
//        uBuffer.get(nv21, ySize + vSize, uSize);
//
//        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
//        ByteArrayOutputStream out = new ByteArrayOutputStream();
//        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 90, out);
//        byte[] imageBytes = out.toByteArray();
//
//        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
//    }



//    private Bitmap yuv420ToBitmap(ByteBuffer yBuffer, ByteBuffer uBuffer, ByteBuffer vBuffer, int width, int height) {
//        int ySize = yBuffer.remaining();
//        int uSize = uBuffer.remaining();
//        int vSize = vBuffer.remaining();
//
//        byte[] nv21 = new byte[ySize + uSize + vSize * 2];
//
//        yBuffer.get(nv21, 0, ySize);
//
//        for (int i = 0; i < uSize; i++) {
//            nv21[ySize + (i * 2)] = vBuffer.get(i);
//            nv21[ySize + (i * 2) + 1] = uBuffer.get(i);
//        }
//
//        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, width, height, null);
//        ByteArrayOutputStream out = new ByteArrayOutputStream();
//        yuvImage.compressToJpeg(new Rect(0, 0, width, height), 100, out);
//
//        byte[] imageBytes = out.toByteArray();
//        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
//    }
//
//    private Bitmap imageProxyToBitmap(ImageProxy image) {
//        @SuppressLint("UnsafeOptInUsageError")
//        Image.Plane[] planes = image.getImage().getPlanes();
//        ByteBuffer yBuffer = planes[0].getBuffer();
//        ByteBuffer uBuffer = planes[1].getBuffer();
//        ByteBuffer vBuffer = planes[2].getBuffer();
//
//
//
//        return yuv420ToBitmap(yBuffer, uBuffer, vBuffer, image.getWidth(), image.getHeight());
//    }

//    private Bitmap imageProxyToBitmap(ImageProxy image, Context context) {
//        if (image.getFormat() != ImageFormat.YUV_420_888) {
//            throw new IllegalArgumentException("Unsupported format");
//        }
//
//        RenderScript rs = RenderScript.create(context);
//        ScriptIntrinsicYuvToRGB yuvToRgbScript = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
//
//        Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(image.getWidth()).setY(image.getHeight()).setYuvFormat(ImageFormat.YUV_420_888);
//        Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);
//
//        Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(image.getWidth()).setY(image.getHeight());
//        Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
//
//        in.copyFrom(getYuvBytes(image));
//
//        yuvToRgbScript.setInput(in);
//        yuvToRgbScript.forEach(out);
//
//        Bitmap bitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
//        out.copyTo(bitmap);
//
//        rs.destroy();
//
//        return bitmap;
//    }
//
//    private byte[] getYuvBytes(ImageProxy image) {
//        @SuppressLint("UnsafeOptInUsageError")
//        Image.Plane[] planes = image.getImage().getPlanes();
//        byte[] yuvBytes = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
//        int dstIndex = 0;
//
//        for (int i = 0; i < planes.length; i++) {
//            ByteBuffer buffer = planes[i].getBuffer();
//            byte[] bytes = new byte[buffer.remaining()];
//            buffer.get(bytes);
//
//            for (int j = 0; j < bytes.length; j++) {
//                yuvBytes[dstIndex++] = bytes[j];
//            }
//        }
//        return yuvBytes;
//    }


}
