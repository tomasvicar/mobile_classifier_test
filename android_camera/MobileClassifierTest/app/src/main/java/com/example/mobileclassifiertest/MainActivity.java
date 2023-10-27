package com.example.mobileclassifiertest;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

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
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_LOAD_MODEL = 1;
    private static final int REQUEST_LOAD_CLASSES = 2;
    private static final int REQUEST_LOAD_IMAGE = 3;

    private Button btnLoadModel;
    private Button btnLoadClasses;
    private Button btnLoadImage;
    private ImageView imagePreview;
    private TextView tvResult;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnLoadModel = findViewById(R.id.btn_load_model);
        btnLoadClasses = findViewById(R.id.btn_load_classes);
        btnLoadImage = findViewById(R.id.btn_load_image);
        imagePreview = findViewById(R.id.image_preview);
        tvResult = findViewById(R.id.tv_result);

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
            // Logic to load Image to Classify
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, REQUEST_LOAD_IMAGE);
        });

        File file = new File(this.getFilesDir(),"model.ptl");
        boolean result = file.delete();

        File file2 = new File(this.getFilesDir(),"classes.ptl");
        boolean result2 = file.delete();

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
                        processImage(bitmap);

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    break;
            }
        }
    }

    private void processImage(Bitmap bitmap) {



        List<String> classes = LoadClasses("classes.txt");
        LoadTorchModule("model.ptl");

//
        float[] mean_norm = TensorImageUtils.TORCHVISION_NORM_MEAN_RGB;
        float[] std_norm = TensorImageUtils.TORCHVISION_NORM_STD_RGB;

        // Resize Bitmap to 256x256
        Bitmap resizedBitmap = resizeBitmapTo256x256(bitmap);

        // Center Crop to 224x224
        Bitmap croppedBitmap = centerCropTo224x224(resizedBitmap);
        imagePreview.setImageBitmap(croppedBitmap);

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

//        String path = assetFilePath(this, "model.ptl");

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
//            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open(fileName)));
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


    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}