namespace Nncase.TIR
{
    public enum BufferMode
    {
        Default,
        // Maps buffer[i][j][k] -> buffer[i][0][k] if dimension i's shape equals 1.
        AutoBroadcast,
    }
}