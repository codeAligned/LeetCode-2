public class Solution {
    public int singleNumber(int[] A) {
    	int res = 0;
    	HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
    	for(int i = 0; i < A.length;i++){
    		if(!map.containsKey(A[i]))
    			map.put(A[i], 1);
    		else map.put(A[i], 2);
    	}
    
    	for(Object key:map.keySet()){
    		if(map.get(key) == 1)
    			res = Integer.valueOf(key.toString());
    	}
    	
    	return res;
    }
}