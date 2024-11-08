import React, { useState } from "react";
import { View, TextInput, Button } from "react-native";
import { register } from "../(services)/api/api";
import { RegisterScreenProps } from "../../type";

const RegisterScreen: React.FC<RegisterScreenProps> = ({ navigation }) => {
  const [username, setUsername] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");

  const handleRegister = async () => {
    try {
      await register(username, email, password);
      navigation.navigate("Login");
    } catch (err) {
      console.error("Error registering user", err);
    }
  };

  return (
    <View>
      <TextInput
        placeholder="Username"
        onChangeText={setUsername}
        value={username}
      />
      <TextInput placeholder="Email" onChangeText={setEmail} value={email} />
      <TextInput
        placeholder="Password"
        secureTextEntry
        onChangeText={setPassword}
        value={password}
      />
      <Button title="Register" onPress={handleRegister} />
    </View>
  );
};

export default RegisterScreen;
