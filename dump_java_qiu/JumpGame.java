public class Solution {
    public boolean canJump(int[] A) {
        if (A == null || A.length == 0) {
            return true;
        }
        int maxReach = 0;
        
        for (int start = 0; start < A.length && start <= maxReach; start++) {
            if (A[start] + start > maxReach) {
                maxReach = A[start] + start;
            }
            if (maxReach >= A.length - 1) {
                return true;
            }
        }
        return false;
    }
}
