using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Polls the EMG ui_server.py for the current movement state.
/// Attach to any GameObject. Hook into OnMovementChanged to drive your VR arm.
///
/// Setup:
///   1. Drop this script on a GameObject in your scene
///   2. Subscribe to OnMovementChanged from your arm controller script
///   3. Run ui_server.py on the same machine (or set serverUrl to the right IP)
///
/// The server returns: {"movement": "idle|clench|wrist_extension", "strength": 52.3}
/// </summary>
public class EMGReceiver : MonoBehaviour
{
    [Header("Server")]
    [Tooltip("URL of the ui_server.py /state endpoint")]
    public string serverUrl = "http://localhost:8080/state";

    [Tooltip("How often to poll (seconds)")]
    public float pollInterval = 0.3f;

    [Header("State (read-only)")]
    public string currentMovement = "idle";
    public float currentStrength = 0f;

    /// <summary>
    /// Fired whenever the movement changes. Args: (newMovement, strength)
    /// </summary>
    public static event Action<string, float> OnMovementChanged;

    /// <summary>
    /// Fired every poll with the latest data. Args: (movement, strength)
    /// </summary>
    public static event Action<string, float> OnStateUpdate;

    private string _lastMovement = "idle";

    [Serializable]
    private class EMGState
    {
        public string movement;
        public float strength;
    }

    void Start()
    {
        StartCoroutine(PollLoop());
    }

    IEnumerator PollLoop()
    {
        while (true)
        {
            using (UnityWebRequest req = UnityWebRequest.Get(serverUrl))
            {
                req.timeout = 2;
                yield return req.SendWebRequest();

                if (req.result == UnityWebRequest.Result.Success)
                {
                    EMGState state = JsonUtility.FromJson<EMGState>(req.downloadHandler.text);
                    currentMovement = state.movement;
                    currentStrength = state.strength;

                    OnStateUpdate?.Invoke(currentMovement, currentStrength);

                    if (currentMovement != _lastMovement)
                    {
                        OnMovementChanged?.Invoke(currentMovement, currentStrength);
                        _lastMovement = currentMovement;
                    }
                }
            }

            yield return new WaitForSeconds(pollInterval);
        }
    }
}
