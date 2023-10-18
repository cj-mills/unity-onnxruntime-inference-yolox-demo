using System;
using System.Runtime.InteropServices;

/// <summary>
/// Class with Native plugin functions for ONNX Runtime.
/// </summary>
public static class ONNXRuntimePlugin
{
    // Name of the DLL file
    const string dll = "UnityONNXInferenceCVPlugin";

    [DllImport(dll)]
    public static extern int InitOrtAPI();

    [DllImport(dll)]
    public static extern int GetProviderCount();

    [DllImport(dll)]
    public static extern IntPtr GetProviderName(int index);

    [DllImport(dll)]
    public static extern void FreeResources();

    [DllImport(dll)]
    public static extern IntPtr LoadModel(string model, string execution_provider, int[] inputDims);

    [DllImport(dll)]
    public static extern void PerformInference(byte[] inputData, float[] outputArray, int length);
}
