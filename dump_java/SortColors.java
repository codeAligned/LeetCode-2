public class Solution {
    public void sortColors(int[] A) {

		int h = 0, t = A.length - 1;
		while (h < A.length && A[h] == 0) {
			h++;
		}

		int cur = h;
		int tmp;
		while (cur <= t) {

			if (A[cur] == 0) {
				if (cur > h) {
					tmp = A[h];
					A[h] = A[cur];
					A[cur] = tmp;
					h++;
				} else {
					cur++;
					h++;
				}
			} else if (A[cur] == 2) {
				if (cur < t) {
					tmp = A[t];
					A[t] = A[cur];
					A[cur] = tmp;
					t--;
				} else {
					return;
				}

			} else {
				cur++;
			}
		}

	}
}