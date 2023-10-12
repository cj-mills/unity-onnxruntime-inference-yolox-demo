#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

[InitializeOnLoad]
public class Startup
{
    static Startup()
    {
        // Get all files named "DirectML.dll" in the Assets directory
        string[] files = Directory.GetFiles("./Assets/", "DirectML.dll", SearchOption.AllDirectories);
        // Iterate through each found file
        foreach (string file in files)
        {
            // Check if the file is in the "x86_64" folder
            if (file.Contains("x86_64"))
            {
                // Get the file path for the Editor application
                string editorPath = EditorApplication.applicationPath;
                // Extract the parent folder for the Editor application
                string editorDir = Directory.GetParent(editorPath).ToString();
                // Define target file path
                string targetPath = $"{editorDir}/DirectML.dll";
                // Only copy the file to the Editor application folder if it is not already present
                if (!File.Exists(targetPath)) File.Copy(file, targetPath);
            }
        }
    }
}
#endif