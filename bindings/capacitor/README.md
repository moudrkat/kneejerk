# capacitor-kneejerk

Decision models inside a Capacitor app, natively. `load` a GGUF, `decide` with a `/v1/systemone`
request, get probabilities back. Android first (arm64), iOS next.

```ts
import { Kneejerk } from 'capacitor-kneejerk'

await Kneejerk.load({ path: modelPath, threads: 4 })            // once; 1.1 GB d1-2B takes a second or two
const { response, wallMs } = await Kneejerk.decide({
  request: JSON.stringify({ state, questions: { kind: { type: 'choice', instructions: '…', criteria: { … } } } }),
})
const answers = JSON.parse(response).answers
```

Build: the Android module compiles kneejerk and llama.cpp from source with the NDK
(`externalNativeBuild`, CMake 3.22+, NDK r27), one `libkneejerk_jni.so` for arm64-v8a. `kneejerkRoot`
in `gradle.properties` points at the kneejerk checkout when the plugin is used from outside the repo.

The model file is the app's business: download it to the app's files directory (wifi only, it is
1.1 GB for d1-2B) and pass the path. Pixel 8a: 464 ms per yes/no decision with repack, 2.1 GB RSS.

## Downloading a model

`Kneejerk.download({ url, path })` fetches a model file and continues an interrupted download where it
stopped (HTTP Range, a `.part` file until it is whole). Progress comes as `downloadProgress` events.
Android only for now.
