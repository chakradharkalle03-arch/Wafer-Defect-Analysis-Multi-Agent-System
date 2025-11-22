# HuggingFace Open Source Models Used

## Models Integrated

### 1. Object Detection
- **Model**: `facebook/detr-resnet-50`
- **Type**: DETR (Detection Transformer)
- **License**: Apache 2.0
- **Use**: Detects defects in wafer images
- **API Endpoint**: `https://api-inference.huggingface.co/models/facebook/detr-resnet-50`

### 2. Image Classification
- **Model**: `google/vit-base-patch16-224`
- **Type**: Vision Transformer (ViT)
- **License**: Apache 2.0
- **Use**: Classifies defect types and extracts features
- **API Endpoint**: `https://api-inference.huggingface.co/models/google/vit-base-patch16-224`

## How It Works

The system uses **HuggingFace Inference API** which:
- ✅ No local PyTorch installation required
- ✅ No DLL dependencies
- ✅ Uses open source models
- ✅ Works via HTTP API calls
- ✅ Automatic model loading on HuggingFace servers

## Fallback Strategy

1. **First**: Try to use local models (if PyTorch available)
2. **Second**: Use HuggingFace Inference API
3. **Third**: Use custom detection algorithms (always available)

## API Usage

All models are accessed via:
```
POST https://api-inference.huggingface.co/models/{model_name}
Headers: Authorization: Bearer {your_hf_token}
Body: Raw image bytes (PNG format)
```

## Benefits

- **No Local Dependencies**: Works without PyTorch
- **Always Updated**: Uses latest model versions from HuggingFace
- **Scalable**: API handles model loading and inference
- **Open Source**: All models are open source and free to use
- **Reliable**: HuggingFace infrastructure handles scaling

## Model Information

### DETR (Detection Transformer)
- **Paper**: End-to-End Object Detection with Transformers
- **Architecture**: Transformer-based object detection
- **Performance**: State-of-the-art object detection
- **Input**: Images of any size
- **Output**: Bounding boxes with confidence scores

### ViT (Vision Transformer)
- **Paper**: An Image is Worth 16x16 Words
- **Architecture**: Transformer-based image classification
- **Performance**: Excellent for image classification
- **Input**: 224x224 images (automatically resized)
- **Output**: Classification scores

## Rate Limits

HuggingFace Inference API has rate limits based on your account tier:
- **Free tier**: Limited requests per minute
- **Pro tier**: Higher limits
- **Enterprise**: Custom limits

The system includes error handling for rate limits and will fall back to custom algorithms if needed.

