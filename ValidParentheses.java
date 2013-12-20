/**
 * Valid Parentheses Total Accepted: 3635 Total Submissions: 13252 My Submissions
 * Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
 * 
 * The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
**/
public class Solution {
    public boolean isValid(String s) {
		HashMap<Character,Character> map = new HashMap<Character,Character>();
		map.put('(', ')');
		map.put('[', ']');
		map.put('{', '}');

		Stack<Character> ss = new Stack<Character>();
		char tmp;

		for (int i = 0; i < s.length(); i++) {
			tmp = s.charAt(i);
			
			if(map.containsKey(tmp)) {
				ss.push(tmp);
			}else if(map.containsValue(tmp)) {
				if(ss.empty() || !(map.get(ss.pop())==tmp)){
					return false;
				}
			}else {
				return false;
			}
		}

		if (ss.empty())
			return true;

		return false;
    }
}