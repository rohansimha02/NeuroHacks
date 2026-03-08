using UnityEngine;

/// <summary>
/// Example: drives a VR arm Animator based on EMG movement state.
/// Attach to the same GameObject as your Animator, or reference it.
///
/// Expects Animator parameters:
///   - "Clench" (bool)   — triggers fist clench animation
///   - "Extend" (bool)   — triggers wrist extension animation
///
/// If you're not using Animator, replace the body of HandleMovement()
/// with whatever drives your arm (blend shapes, IK targets, etc).
/// </summary>
public class EMGArmController : MonoBehaviour
{
    [Header("References")]
    public Animator armAnimator;

    void OnEnable()
    {
        EMGReceiver.OnMovementChanged += HandleMovement;
    }

    void OnDisable()
    {
        EMGReceiver.OnMovementChanged -= HandleMovement;
    }

    void HandleMovement(string movement, float strength)
    {
        if (armAnimator == null) return;

        // Reset all
        armAnimator.SetBool("Clench", false);
        armAnimator.SetBool("Extend", false);

        switch (movement)
        {
            case "clench":
                armAnimator.SetBool("Clench", true);
                Debug.Log($"[EMG] CLENCH (strength={strength:F0})");
                break;

            case "wrist_extension":
                armAnimator.SetBool("Extend", true);
                Debug.Log($"[EMG] WRIST EXTENSION (strength={strength:F0})");
                break;

            default: // idle
                Debug.Log($"[EMG] IDLE (strength={strength:F0})");
                break;
        }
    }
}
