import { View } from "react-native";

import { Link } from "expo-router";

export default function Page() {
  return (
    <View>
      <Link href="./auth/register">Register</Link>
      {/* ...other links */}
      <Link href="./auth/login">Login</Link>
    </View>
  );
}
