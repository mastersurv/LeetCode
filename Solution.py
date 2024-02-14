class Solution:
	def romanToInt(self, s: str) -> int:
		alphabet = {
			'I': 1,
			'V': 5,
			'X': 10,
			'L': 50,
			'C': 100,
			'D': 500,
			'M': 1000
		}
		res = 0
		flag = False
		for i in range(len(s) - 1):
			if flag:
				flag = False
				continue
			if s[i] == 'I' and s[i + 1] in ('V', 'X'):
				res += alphabet[s[i + 1]] - 1
				flag = True
			elif s[i] == 'X' and s[i + 1] in ('L', 'C'):
				res += alphabet[s[i + 1]] - 10
				flag = True
			elif s[i] == 'C' and s[i + 1] in ('D', 'M'):
				res += alphabet[s[i + 1]] - 100
				flag = True
			else:
				res += alphabet[s[i]]
		if len(s) != 1:
			i += 1
			if not flag:
				res += alphabet[s[i]]
		else:
			res = alphabet[s[0]]
		return res

	def isPalindrome(self, x: int) -> bool:
		x = str(x)
		if x[::-1] == x:
			return True
		return False

	def twoSum(self, nums: list[int], target: int) -> list[int]:
		answer = []
		ind1 = 0
		ind2 = 0
		for i in range(len(nums) - 1):
			for j in range(i + 1, len(nums)):
				if nums[i] + nums[j] == target:
					answer.append(i)
					answer.append(j)
					return answer

		return answer


# if __name__ == '__main__':
# 	s = Solution()
# 	res = s.romanToInt('I')
# 	# res = s.twoSum([3,3], 6)
# 	print(res)
