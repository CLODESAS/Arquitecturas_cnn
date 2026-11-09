# Diagramas de las arquitecturas CNN (CIFAR-10)

Diagramas de flujo de las tres arquitecturas probadas en [Arquitecturas_cnn.ipynb](Arquitecturas_cnn.ipynb).

## Dataset y salida del modelo

**Imágenes:** CIFAR-10 tiene 60,000 imágenes a color (RGB) de 32x32 píxeles, 50,000 para entrenamiento y 10,000 para prueba. Son fotos pequeñas de objetos y animales comunes, de baja resolución.

**Qué predecimos:** a qué clase pertenece cada imagen. Hay 10 clases: avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

**Salida del modelo:** un vector de 10 probabilidades, una por clase, que suman 1 (capa `Dense(10, softmax)`). La clase con mayor probabilidad es la predicción. Por ejemplo, si la salida más alta es la posición 3, el modelo dice "gato".

Es un problema de clasificación multiclase. Las etiquetas son números del 0 al 9, por eso se usó `sparse_categorical_crossentropy`.

## 1. CNN simple (`cnn_simple`)

```mermaid
flowchart TD
    A["Entrada 32x32x3"] --> B["Data augmentation<br/>flip, rotación, zoom, traslación"]
    B --> C["Rescaling 1/255"]
    C --> D["Conv2D 32, 3x3, ReLU"]
    D --> E["MaxPooling 2x2<br/>16x16x32"]
    E --> F["Conv2D 64, 3x3, ReLU"]
    F --> G["MaxPooling 2x2<br/>8x8x64"]
    G --> H["Flatten<br/>4096"]
    H --> I["Dense 128, ReLU"]
    I --> J["Dropout 0.5"]
    J --> K["Dense 10, Softmax"]
    K --> L["Probabilidad por clase"]
```

## 2. CNN profunda (`cnn_profunda`)

```mermaid
flowchart TD
    A["Entrada 32x32x3"] --> B["Data augmentation"]
    B --> C["Rescaling 1/255"]
    subgraph B1["Bloque 1 - 32 filtros"]
        D1["Conv2D 32 + L2"] --> D2["BatchNorm + ReLU"] --> D3["Conv2D 32 + L2"] --> D4["BatchNorm + ReLU"] --> D5["MaxPooling 16x16"] --> D6["Dropout 0.2"]
    end
    subgraph B2["Bloque 2 - 64 filtros"]
        E1["Conv2D 64 + L2"] --> E2["BatchNorm + ReLU"] --> E3["Conv2D 64 + L2"] --> E4["BatchNorm + ReLU"] --> E5["MaxPooling 8x8"] --> E6["Dropout 0.3"]
    end
    subgraph B3["Bloque 3 - 128 filtros"]
        F1["Conv2D 128 + L2"] --> F2["BatchNorm + ReLU"] --> F3["Conv2D 128 + L2"] --> F4["BatchNorm + ReLU"] --> F5["MaxPooling 4x4"] --> F6["Dropout 0.4"]
    end
    C --> D1
    D6 --> E1
    E6 --> F1
    F6 --> G["GlobalAveragePooling2D<br/>128"]
    G --> H["Dense 128, ReLU + L2"]
    H --> I["Dropout 0.5"]
    I --> J["Dense 10, Softmax"]
```

## 3. MobileNetV2 con transfer learning (`mobilenetv2_transfer`)

```mermaid
flowchart TD
    A["Entrada 32x32x3"] --> B["Data augmentation"]
    B --> C["Resizing 96x96"]
    C --> D["preprocess_input<br/>rango -1 a 1"]
    D --> E["MobileNetV2 preentrenada<br/>ImageNet, congelada<br/>salida 3x3x1280"]
    E --> F["GlobalAveragePooling2D<br/>1280"]
    F --> G["Dropout 0.3"]
    G --> H["Dense 10, Softmax"]
    H --> I["Probabilidad por clase"]
    style E fill:#cfe8ff,stroke:#336
    style H fill:#ffe9b3,stroke:#963
```

Azul: pesos congelados. Naranja: única parte que se entrena.
