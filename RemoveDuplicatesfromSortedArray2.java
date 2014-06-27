public class Solution {
    public int removeDuplicates(int[] A) {
        
		if(A.length <= 2)
			return A.length;
		
		int cur = 2; 
		int pre = 1;
		
		while(cur < A.length){
			if(A[cur] == A[pre] && A[cur] == A[pre-1]){
				cur++;
			}else{
				pre++;
				A[pre]=A[cur];
				cur++;
			}
		}
		
		return pre+1;
	
    }
}