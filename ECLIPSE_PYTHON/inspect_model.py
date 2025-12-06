import tensorflow as tf
import traceback

print("\n=== STARTING MODEL INSPECTION ===")

try:
    print("\nLoading model...")
    model = tf.keras.models.load_model("models/densenet169_unet_final.h5", compile=False)
    print("\nMODEL LOADED SUCCESSFULLY!")

    print("\n=== MODEL INPUT SHAPE ===")
    print(model.input_shape)

    print("\n=== MODEL OUTPUT SHAPE ===")
    print(model.output_shape)

    print("\n=== MODEL SUMMARY (LAST 20 LINES) ===")
    model.summary()

except Exception as e:
    print("\n\n=== ERROR OCCURRED ===")
    print(str(e))
    print("\nFULL TRACEBACK:")
    traceback.print_exc()
